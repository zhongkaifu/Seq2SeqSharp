

using AdvUtils;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{
    public class RoundArray<T>
    {
        private readonly T[] m_array;
        private int currentIdx = 0;
        public RoundArray(T[] a)
        {
            m_array = a;
        }

        public T GetNextItem()
        {
            T item = m_array[currentIdx];
            currentIdx = (currentIdx + 1) % m_array.Length;

            return item;
        }
    }

    public class AttentionSeq2Seq : BaseSeq2SeqFramework
    {
        private readonly Seq2SeqModelMetaData m_modelMetaData;

        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IWeightTensor> m_sharedEmbedding; //The embeddings over devices for both source and target

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The LSTM decoders over devices

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;

        // optimization  hyperparameters
        private readonly float m_dropoutRatio = 0.0f;
        private readonly int m_beamSearchSize = 1;

        private readonly int m_maxSrcSntSize = 128;
        private readonly int m_maxTgtSntSize = 128;

        private readonly bool m_isSrcEmbTrainable = true;
        private readonly bool m_isTgtEmbTrainable = true;
        private readonly bool m_isEncoderTrainable = true;
        private readonly bool m_isDecoderTrainable = true;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;

        public AttentionSeq2Seq(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f, 
            bool isSrcEmbTrainable = true, bool isTgtEmbTrainable = true, bool isEncoderTrainable = true, bool isDecoderTrainable = true, 
            int maxSrcSntSize = 128, int maxTgtSntSize = 128, float memoryUsageRatio = 0.9f, ShuffleEnums shuffleType = ShuffleEnums.Random, string[] compilerOptions = null, int beamSearchSize = 1)
            : base(deviceIds, processorType, modelFilePath, memoryUsageRatio, compilerOptions)
        {
            m_dropoutRatio = dropoutRatio;
            m_isSrcEmbTrainable = isSrcEmbTrainable;
            m_isTgtEmbTrainable = isTgtEmbTrainable;
            m_isEncoderTrainable = isEncoderTrainable;
            m_isDecoderTrainable = isDecoderTrainable;
            m_maxSrcSntSize = maxSrcSntSize;
            m_maxTgtSntSize = maxTgtSntSize;
            m_shuffleType = shuffleType;
            m_beamSearchSize = beamSearchSize;

            m_modelMetaData = LoadModel(CreateTrainableParameters) as Seq2SeqModelMetaData;
        }

        public AttentionSeq2Seq(int srcEmbeddingDim, int tgtEmbeddingDim, int hiddenDim, int encoderLayerDepth, int decoderLayerDepth, Vocab vocab, string srcEmbeddingFilePath, string tgtEmbeddingFilePath,
            string modelFilePath, float dropoutRatio, int multiHeadNum, ProcessorTypeEnums processorType, EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, bool enableCoverageModel, int[] deviceIds,
            bool isSrcEmbTrainable = true, bool isTgtEmbTrainable = true, bool isEncoderTrainable = true, bool isDecoderTrainable = true, 
            int maxSrcSntSize = 128, int maxTgtSntSize = 128, float memoryUsageRatio = 0.9f, ShuffleEnums shuffleType = ShuffleEnums.Random, string[] compilerOptions = null, bool sharedEmbeddings = false)
            : base(deviceIds, processorType, modelFilePath, memoryUsageRatio, compilerOptions)
        {
            m_modelMetaData = new Seq2SeqModelMetaData(hiddenDim, srcEmbeddingDim, tgtEmbeddingDim, encoderLayerDepth, decoderLayerDepth, multiHeadNum, encoderType, decoderType, vocab, enableCoverageModel, sharedEmbeddings);
            m_dropoutRatio = dropoutRatio;

            m_isSrcEmbTrainable = isSrcEmbTrainable;
            m_isTgtEmbTrainable = isTgtEmbTrainable;
            m_isEncoderTrainable = isEncoderTrainable;
            m_isDecoderTrainable = isDecoderTrainable;
            m_maxSrcSntSize = maxSrcSntSize;
            m_maxTgtSntSize = maxTgtSntSize;
            m_shuffleType = shuffleType;

            //Initializng weights in encoders and decoders
            CreateTrainableParameters(m_modelMetaData);

            // Load external embedding from files
            for (int i = 0; i < DeviceIds.Length; i++)
            {
                //If pre-trained embedding weights are speicifed, loading them from files
                if (!string.IsNullOrEmpty(srcEmbeddingFilePath))
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{srcEmbeddingFilePath}' for source side.");
                    LoadWordEmbedding(srcEmbeddingFilePath, m_srcEmbedding.GetNetworkOnDevice(i), m_modelMetaData.Vocab.SrcWordToIndex);
                }

                if (!string.IsNullOrEmpty(tgtEmbeddingFilePath))
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{tgtEmbeddingFilePath}' for target side.");
                    LoadWordEmbedding(tgtEmbeddingFilePath, m_tgtEmbedding.GetNetworkOnDevice(i), m_modelMetaData.Vocab.TgtWordToIndex);
                }
            }
        }

        private bool CreateTrainableParameters(IModelMetaData mmd)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            Seq2SeqModelMetaData modelMetaData = mmd as Seq2SeqModelMetaData;
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim = 0;
            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.SrcEmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: m_isEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.SrcEmbeddingDim, modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(), 
                    isTrainable: m_isEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim;
            }

            if (modelMetaData.DecoderType == DecoderTypeEnums.AttentionLSTM)
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                     new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.TgtEmbeddingDim, contextDim,
                     modelMetaData.Vocab.TargetWordSize, m_dropoutRatio, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem(), modelMetaData.EnableCoverageModel, isTrainable: m_isDecoderTrainable), DeviceIds);
            }
            else
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new TransformerDecoder("TransformerDecoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.TgtEmbeddingDim, modelMetaData.Vocab.TargetWordSize, modelMetaData.DecoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: m_isDecoderTrainable), DeviceIds);
            }

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer || modelMetaData.DecoderType == DecoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(Math.Max(m_maxSrcSntSize, m_maxTgtSntSize) + 2, contextDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), DeviceIds, true);
            }
            else
            {
                m_posEmbedding = null;
            }

            if (modelMetaData.SharedEmbeddings)
            {
                Logger.WriteLine($"Creating shared embeddings for both source side and target side. Shape = '({modelMetaData.Vocab.SourceWordSize} ,{modelMetaData.SrcEmbeddingDim})'");
                m_sharedEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.SrcEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SharedEmbeddings", isTrainable: m_isSrcEmbTrainable), DeviceIds);

                m_srcEmbedding = null;
                m_tgtEmbedding = null;
            }
            else
            {
                Logger.WriteLine($"Creating embeddings for source side. Shape = '({modelMetaData.Vocab.SourceWordSize} ,{modelMetaData.SrcEmbeddingDim})'");
                m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.SrcEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_isSrcEmbTrainable), DeviceIds);

                Logger.WriteLine($"Creating embeddings for target side. Shape = '({modelMetaData.Vocab.TargetWordSize} ,{modelMetaData.TgtEmbeddingDim})'");
                m_tgtEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.TargetWordSize, modelMetaData.TgtEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: m_isTgtEmbTrainable), DeviceIds);

                m_sharedEmbedding = null;
            }

            return true;
        }



        private void LoadWordEmbedding(string extEmbeddingFilePath, IWeightTensor embeddingMatrix, IEnumerable<KeyValuePair<string, int>> wordToIndex)
        {
            Txt2Vec.Model extEmbeddingModel = new Txt2Vec.Model();

            if (extEmbeddingFilePath.EndsWith("txt", StringComparison.InvariantCultureIgnoreCase))
            {
                extEmbeddingModel.LoadTextModel(extEmbeddingFilePath);
            }
            else
            {
                extEmbeddingModel.LoadBinaryModel(extEmbeddingFilePath);
            }

            if (extEmbeddingModel.VectorSize != embeddingMatrix.Columns)
            {
                throw new ArgumentException($"Inconsistent embedding size. ExtEmbeddingModel size = '{extEmbeddingModel.VectorSize}', EmbeddingMatrix column size = '{embeddingMatrix.Columns}'");
            }

            foreach (KeyValuePair<string, int> pair in wordToIndex)
            {
                float[] vector = extEmbeddingModel.GetVector(pair.Key);
                if (vector != null)
                {
                    embeddingMatrix.SetWeightAtRow(pair.Value, vector);
                }
            }
        }

        public void Train(int maxTrainingEpoch, ParallelCorpus trainCorpus, ParallelCorpus validCorpus, ILearningRate learningRate, List<IMetric> metrics, IOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, metrics, m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        public void Valid(ParallelCorpus validCorpus, List<IMetric> metrics)
        {
            RunValid(validCorpus, RunForwardOnSingleDevice, metrics, true);
        }

        public List<List<List<string>>> Test(List<List<string>> inputTokens)
        {
            return RunTest(inputTokens, RunForwardOnSingleDevice);
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IDecoder, IWeightTensor, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), 
                    m_decoder.GetNetworkOnDevice(deviceIdIdx), 
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx), 
                    m_posEmbedding == null ? null : m_posEmbedding.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private NetworkResult RunForwardOnSingleDevice(IComputeGraph computeGraph, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {
            NetworkResult nr = new NetworkResult();

            (IEncoder encoder, IDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor posEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);
            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            List<int> originalSrcLengths = ParallelCorpus.PadSentences(srcSnts);
            int srcSeqPaddedLen = srcSnts[0].Count;
            IWeightTensor srcSelfMask = m_shuffleType == ShuffleEnums.NoPaddingInSrc ? null : computeGraph.BuildPadSelfMask(srcSeqPaddedLen, originalSrcLengths); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            var srcTokensList = m_modelMetaData.Vocab.GetSourceWordIndex(srcSnts);
            IWeightTensor encOutput = Encode(computeGraph, srcTokensList, encoder, srcEmbedding, srcSelfMask, posEmbedding, originalSrcLengths);

            // Generate output decoder sentences
            var tgtTokensList = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts);

            if (decoder is AttentionDecoder)
            {
                nr.Cost = DecodeAttentionLSTM(tgtTokensList, computeGraph, encOutput, decoder as AttentionDecoder, tgtEmbedding, srcSnts.Count, isTraining);
                nr.Beam2Batch2Output = new List<List<List<string>>>();
                nr.Beam2Batch2Output.Add(m_modelMetaData.Vocab.ConvertTargetIdsToString(tgtTokensList));
            }
            else
            {
                if (isTraining)
                {
                    (var c, var tmp) = DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, tgtEmbedding, posEmbedding, DeviceIds[deviceIdIdx], originalSrcLengths, isTraining);
                    nr.Cost = c;
                    nr.Beam2Batch2Output = null;
                }
                else
                {
                    List<List<List<int>>> beam2batch2tgtTokens = new List<List<List<int>>>(); // (beam_search_size, batch_size, tgt_token_size)
                    beam2batch2tgtTokens.Add(tgtTokensList);
                    int batchSize = srcSnts.Count;
                    for (int i = 0; i < m_maxTgtSntSize; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = new List<List<BeamSearchStatus>>(); //(batch_size, beam_search_size)
                        for (int j = 0; j < batchSize; j++)
                        {
                            batch2beam2seq.Add(new List<BeamSearchStatus>());
                        }

                        foreach (var batch2tgtTokens in beam2batch2tgtTokens)
                        {
                            using (var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}"))
                            {
                                (var cost2, var bssSeqList) = DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, tgtEmbedding, posEmbedding, DeviceIds[deviceIdIdx], originalSrcLengths, isTraining, beamSearchSize: m_beamSearchSize);

                                for (int j = 0; j < m_beamSearchSize; j++)
                                {
                                    for (int k = 0; k < batchSize; k++)
                                    {
                                        batch2beam2seq[k].Add(bssSeqList[j][k]);
                                    }
                                }
                            }

                        }

                        // Keep top N result and drop all others
                        for (int k = 0; k < batchSize; k++)
                        {
                            batch2beam2seq[k] = BeamSearch.GetTopNBSS(batch2beam2seq[k], m_beamSearchSize);
                        }

                        beam2batch2tgtTokens.Clear();
                        for (int k = 0; k < m_beamSearchSize; k++)
                        {
                            beam2batch2tgtTokens.Add(new List<List<int>>());
                        }

                        // Convert shape from (batch, beam, seq) to (beam, batch, seq), and check if all output sentences are ended. If so, we will stop decoding.
                        bool allSntsEnd = true;
                        for (int j = 0; j < batchSize; j++)
                        {
                            for (int k = 0; k < m_beamSearchSize; k++)
                            {
                                beam2batch2tgtTokens[k].Add(batch2beam2seq[j][k].OutputIds);

                                if (batch2beam2seq[j][k].OutputIds[batch2beam2seq[j][k].OutputIds.Count - 1] != (int)SENTTAGS.END)
                                {
                                    allSntsEnd = false;
                                }
                            }
                        }
                        if (allSntsEnd)
                        {
                            break;
                        }
                    }

                    nr.Cost = 0.0f;
                    nr.Beam2Batch2Output = m_modelMetaData.Vocab.ConvertTargetIdsToString(beam2batch2tgtTokens);
                }
            }

            nr.RemoveDuplicatedEOS();

            return nr;
        }


        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        /// <param name="g"></param>
        /// <param name="seqs"></param>
        /// <param name="encoder"></param>
        /// <param name="reversEncoder"></param>
        /// <param name="embeddings"></param>
        /// <returns></returns>
        private IWeightTensor Encode(IComputeGraph g, List<List<int>> seqs, IEncoder encoder, IWeightTensor embeddings, IWeightTensor selfMask, IWeightTensor posEmbeddings, List<int> seqOriginalLengths)
        {
         //   int seqLen = srcSeqs[0].Count;
            int batchSize = seqs.Count;

            //List<IWeightTensor> inputs = new List<IWeightTensor>();

            //// Generate batch-first based input embeddings
            //for (int j = 0; j < batchSize; j++)
            //{
            //    int originalLength = originalSrcLengths[j];
            //    for (int i = 0; i < seqLen; i++)
            //    {
            //        var emb = g.PeekRow(Embedding, srcSeqs[j][i], runGradients: i < originalLength ? true : false);

            //        inputs.Add(emb);
            //    }
            //}

            //var inputEmbs = g.ConcatRows(inputs);


            var inputEmbs = ExtractTokensEmbeddings(seqs, g, embeddings, seqOriginalLengths);


            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, m_dropoutRatio);
            }

            return encoder.Encode(inputEmbs, batchSize, g, selfMask);
        }



        private (float, List<List<BeamSearchStatus>>) DecodeTransformer(List<List<int>> tgtSeqs, IComputeGraph g, IWeightTensor encOutputs, TransformerDecoder decoder,
            IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, int deviceId, List<int> srcOriginalLenghts, bool isTraining = true, int beamSearchSize = 1)
        {
            int eosTokenId = m_modelMetaData.Vocab.GetTargetWordIndex(ParallelCorpus.EOS, logUnk: true);
            float cost = 0.0f;

            int batchSize = tgtSeqs.Count;
            var tgtOriginalLengths = ParallelCorpus.PadSentences(tgtSeqs, eosTokenId);
            int tgtSeqLen = tgtSeqs[0].Count;
            int srcSeqLen = encOutputs.Rows / batchSize;

            IWeightTensor srcTgtMask = g.BuildSrcTgtMask(srcSeqLen, tgtSeqLen, tgtOriginalLengths, srcOriginalLenghts);
            IWeightTensor tgtSelfTriMask = g.BuildPadSelfTriMask(tgtSeqLen, tgtOriginalLengths);

            IWeightTensor inputEmbs = ExtractTokensEmbeddings(tgtSeqs, g, tgtEmbedding, tgtOriginalLengths);
            inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbedding, batchSize, inputEmbs, m_dropoutRatio);

            IWeightTensor decOutput = decoder.Decode(inputEmbs, encOutputs, tgtSelfTriMask, srcTgtMask, batchSize, g);
            IWeightTensor probs = g.Softmax(decOutput, runGradients: false, inPlace: true);

            if (isTraining)
            {
                var leftShiftTgtSeqs = ParallelCorpus.LeftShiftSnts(tgtSeqs, eosTokenId);
                var scatterIdxTensor = g.BuildTensorFrom2DArray(leftShiftTgtSeqs, new long[] { leftShiftTgtSeqs.Count * leftShiftTgtSeqs[0].Count, 1 });
                var gatherTensor = g.Gather(probs, scatterIdxTensor, 1);

                var rnd = new TensorSharp.RandomGenerator();
                int idx = rnd.NextSeed() % (batchSize * tgtSeqLen);
                float score = gatherTensor.GetWeightAt(new long[] { idx, 0 });
                cost += (float)-Math.Log(score);

                var lossTensor = g.Add(gatherTensor, -1.0f, false);
                TensorUtils.Scatter(probs, lossTensor, scatterIdxTensor, 1);

                decOutput.CopyWeightsToGradients(probs);

                return (cost, null);
            }
            else
            {
                // Transformer decoder with beam search at inference time
                List<List<BeamSearchStatus>> bssSeqList = new List<List<BeamSearchStatus>>();
                while (beamSearchSize > 0)
                {
                    // Output "i"th target word
                    int[] targetIdx = g.Argmax(probs, 1);

                    List<BeamSearchStatus> outputTgtSeqs = new List<BeamSearchStatus>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        BeamSearchStatus bss = new BeamSearchStatus();
                        bss.OutputIds.AddRange(tgtSeqs[i]);
                        bss.OutputIds.Add(targetIdx[i * tgtSeqLen + tgtSeqLen - 1]);

                        for (int j = 0; j < tgtSeqLen; j++)
                        {
                            var score = probs.GetWeightAt(new long[] { i * tgtSeqLen + j, targetIdx[i * tgtSeqLen + j] });
                            bss.Score += (float)(-Math.Log(score));
                        }

                        outputTgtSeqs.Add(bss);
                    }

                    bssSeqList.Add(outputTgtSeqs);

                    beamSearchSize--;

                    if (beamSearchSize > 0)
                    {
                        for (int i = 0; i < batchSize; i++)
                        {
                            for (int j = 0; j < tgtSeqLen; j++)
                            {
                                probs.SetWeightAt(0.0f, new long[] { i * tgtSeqLen + j, targetIdx[i * tgtSeqLen + j] });
                            }
                        }
                    }
                }
                return (0.0f, bssSeqList);
            }
        }

        private static IWeightTensor ExtractTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, List<int> seqOriginalLengths)
        {
            int batchSize = seqs.Count;
            int seqLen = seqs[0].Count;

            List<IWeightTensor> inputs = new List<IWeightTensor>();
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    var emb = g.Peek(embeddingsTensor, 0, seqs[i][j], runGradients: j < seqOriginalLengths[i] ? true : false);
                    inputs.Add(emb);
                }
            }
            IWeightTensor inputEmbs = inputs.Count > 1 ? g.ConcatRows(inputs) : inputs[0];
            return inputEmbs;
        }


        /// <summary>
        /// Decode output sentences in training
        /// </summary>
        /// <param name="outputSnts">In training mode, they are golden target sentences, otherwise, they are target sentences generated by the decoder</param>
        /// <param name="g"></param>
        /// <param name="encOutputs"></param>
        /// <param name="decoder"></param>
        /// <param name="decoderFFLayer"></param>
        /// <param name="tgtEmbedding"></param>
        /// <returns></returns>
        private float DecodeAttentionLSTM(List<List<int>> outputSnts, IComputeGraph g, IWeightTensor encOutputs, AttentionDecoder decoder, IWeightTensor tgtEmbedding, int batchSize, bool isTraining = true)
        {
            int eosTokenId = m_modelMetaData.Vocab.GetTargetWordIndex(ParallelCorpus.EOS, logUnk: true);
            float cost = 0.0f;
            int[] ix_inputs = new int[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = outputSnts[i][0];
            }

            // Initialize variables accoridng to current mode
            List<int> originalOutputLengths = isTraining ? ParallelCorpus.PadSentences(outputSnts, eosTokenId) : null;
            int seqLen = isTraining ? outputSnts[0].Count : 64;
            float dropoutRatio = isTraining ? m_dropoutRatio : 0.0f;
            HashSet<int> setEndSentId = isTraining ? null : new HashSet<int>();

            // Pre-process for attention model
            AttentionPreProcessResult attPreProcessResult = decoder.PreProcess(encOutputs, batchSize, g);
            for (int i = 1; i < seqLen; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    inputs.Add(g.Peek(tgtEmbedding, 0, ix_inputs[j]));
                }
                IWeightTensor inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                IWeightTensor eOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);

                //Softmax for output
                using (IWeightTensor probs = g.Softmax(eOutput, runGradients: false, inPlace: true))
                {
                    if (isTraining)
                    {
                        //Calculate loss for each word in the batch
                        for (int k = 0; k < batchSize; k++)
                        {
                            float score_k = probs.GetWeightAt(new long[] { k, outputSnts[k][i] });
                            if (i < originalOutputLengths[k])
                            {
                                var lcost = (float)-Math.Log(score_k);
                                if (float.IsNaN(lcost))
                                {
                                    throw new ArithmeticException($"Score = '{score_k}' Cost = Nan at index '{i}' word '{outputSnts[k][i]}', Output Sentence = '{String.Join(" ", outputSnts[k])}'");
                                }
                                else
                                {
                                    cost += lcost;
                                }
                            }

                            probs.SetWeightAt(score_k - 1, new long[] { k, outputSnts[k][i] });
                            ix_inputs[k] = outputSnts[k][i];
                        }
                        eOutput.CopyWeightsToGradients(probs);
                    }
                    else
                    {
                        // Output "i"th target word
                        int[] targetIdx = g.Argmax(probs, 1);
                        for (int j = 0; j < targetIdx.Length; j++)
                        {
                            if (setEndSentId.Contains(j) == false)
                            {
                                outputSnts[j].Add(targetIdx[j]);

                                if (targetIdx[j] == eosTokenId)
                                {
                                    setEndSentId.Add(j);
                                }
                            }
                        }

                        if (setEndSentId.Count == batchSize)
                        {
                            // All target sentences in current batch are finished, so we exit.
                            break;
                        }

                        ix_inputs = targetIdx;
                    }
                }
            }

            return cost;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_modelMetaData.Vocab.DumpSourceVocab(outputSrcVocab);
            m_modelMetaData.Vocab.DumpTargetVocab(outputTgtVocab);
        }
    }
}
