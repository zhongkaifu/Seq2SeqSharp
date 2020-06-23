

using AdvUtils;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The LSTM decoders over devices        

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;

        // optimization  hyperparameters
        private readonly float m_dropoutRatio = 0.0f;
        private readonly int m_defaultDeviceId = 0;
        private readonly int m_maxTgtSntSize = 128;

        private readonly bool m_isSrcEmbTrainable = true;
        private readonly bool m_isTgtEmbTrainable = true;
        private readonly bool m_isEncoderTrainable = true;
        private readonly bool m_isDecoderTrainable = true;

        private readonly bool m_aggregateSrcLength = true;

        public AttentionSeq2Seq(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f, 
            bool isSrcEmbTrainable = true, bool isTgtEmbTrainable = true, bool isEncoderTrainable = true, bool isDecoderTrainable = true, int maxTgtSntSize = 128, float memoryUsageRatio = 0.9f, bool aggregateSrcLength = true)
            : base(deviceIds, processorType, modelFilePath, memoryUsageRatio)
        {
            m_dropoutRatio = dropoutRatio;
            m_isSrcEmbTrainable = isSrcEmbTrainable;
            m_isTgtEmbTrainable = isTgtEmbTrainable;
            m_isEncoderTrainable = isEncoderTrainable;
            m_isDecoderTrainable = isDecoderTrainable;
            m_maxTgtSntSize = maxTgtSntSize;
            m_aggregateSrcLength = aggregateSrcLength;

            m_modelMetaData = LoadModel(CreateTrainableParameters) as Seq2SeqModelMetaData;
        }

        public AttentionSeq2Seq(int embeddingDim, int hiddenDim, int encoderLayerDepth, int decoderLayerDepth, Vocab vocab, string srcEmbeddingFilePath, string tgtEmbeddingFilePath,
            string modelFilePath, float dropoutRatio, int multiHeadNum, ProcessorTypeEnums processorType, EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, bool enableCoverageModel, int[] deviceIds,
            bool isSrcEmbTrainable = true, bool isTgtEmbTrainable = true, bool isEncoderTrainable = true, bool isDecoderTrainable = true, int maxTgtSntSize = 128, float memoryUsageRatio = 0.9f, bool aggregateSrcLength = true)
            : base(deviceIds, processorType, modelFilePath, memoryUsageRatio)
        {
            m_modelMetaData = new Seq2SeqModelMetaData(hiddenDim, embeddingDim, encoderLayerDepth, decoderLayerDepth, multiHeadNum, encoderType, decoderType, vocab, enableCoverageModel);
            m_dropoutRatio = dropoutRatio;

            m_isSrcEmbTrainable = isSrcEmbTrainable;
            m_isTgtEmbTrainable = isTgtEmbTrainable;
            m_isEncoderTrainable = isEncoderTrainable;
            m_isDecoderTrainable = isDecoderTrainable;
            m_maxTgtSntSize = maxTgtSntSize;
            m_aggregateSrcLength = aggregateSrcLength;

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
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: m_isEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(), 
                    isTrainable: m_isEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim;
            }

            if (modelMetaData.DecoderType == DecoderTypeEnums.AttentionLSTM)
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                     new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, contextDim,
                     modelMetaData.Vocab.TargetWordSize, m_dropoutRatio, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem(), modelMetaData.EnableCoverageModel, isTrainable: m_isDecoderTrainable), DeviceIds);
            }
            else
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new TransformerDecoder("TransformerDecoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.Vocab.TargetWordSize, modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: m_isDecoderTrainable), DeviceIds);
            }

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer || modelMetaData.DecoderType == DecoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(BuildPositionWeightTensor(m_maxTgtSntSize + 2, modelMetaData.EmbeddingDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), DeviceIds, true);
            }
            else
            {
                m_posEmbedding = null;
            }

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: NormType.Normal, fanOut: true, name: "SrcEmbeddings", isTrainable: m_isSrcEmbTrainable), DeviceIds);
            m_tgtEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.TargetWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: NormType.Normal, fanOut: true, name: "TgtEmbeddings", isTrainable: m_isTgtEmbTrainable), DeviceIds);

            return true;
        }


        private WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false)
        {
            WeightTensor t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);

            //  double numTimescales = (float)column / 2;
            double logTimescaleIncrement = Math.Log(10000.0f) / (float)(column);
            float[] posWeights = new float[row * column];

            double sqrtC = Math.Sqrt(column);
            for (int p = 0; p < row; ++p)
            {
                for (int i = 0; i < column; i += 2)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));
                    posWeights[p * column + i] = (float)(Math.Sin(v) / sqrtC);
                    posWeights[p * column + i + 1] = (float)(Math.Cos(v) / sqrtC);
                }
            }

            t.TWeight.CopyFrom(posWeights);

            return t;
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

        public void Train(int maxTrainingEpoch, ParallelCorpus trainCorpus, ParallelCorpus validCorpus, ILearningRate learningRate, List<IMetric> metrics, AdamOptimizer optimizer)
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

        public List<List<string>> Test(List<List<string>> inputTokens)
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
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_decoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx), m_posEmbedding == null ? null : m_posEmbedding.GetNetworkOnDevice(deviceIdIdx));
        }

        private void RemoveDuplicatedEOS(List<List<string>> snts)
        {
            foreach (var snt in snts)
            {
                for (int i = 0; i < snt.Count; i++)
                {
                    if (snt[i] == ParallelCorpus.EOS)
                    {
                        snt.RemoveRange(i, snt.Count - i);
                        snt.Add(ParallelCorpus.EOS);
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private float RunForwardOnSingleDevice(IComputeGraph computeGraph, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {
            (IEncoder encoder, IDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor posEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);
            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);


            List<int> originalSrcLengths = ParallelCorpus.PadSentences(srcSnts);
            int srcSeqPaddedLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;
            IWeightTensor srcSelfMask = (m_aggregateSrcLength || m_modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM) ? null 
                                : MaskUtils.BuildPadSelfMask(computeGraph, srcSeqPaddedLen, originalSrcLengths, deviceIdIdx); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            IWeightTensor encOutput = Encode(computeGraph, srcSnts, encoder, srcEmbedding, srcSelfMask, posEmbedding);

            if (srcSelfMask != null)
            {
                srcSelfMask.Dispose();
            }

            // Generate output decoder sentences
            if (decoder is AttentionDecoder)
            {
                return DecodeAttentionLSTM(tgtSnts, computeGraph, encOutput, decoder as AttentionDecoder, tgtEmbedding, srcSnts.Count, isTraining);
            }
            else
            {
                if (isTraining)
                {                    
                    return DecodeTransformer(tgtSnts, computeGraph, encOutput, decoder as TransformerDecoder, tgtEmbedding, posEmbedding, batchSize, deviceIdIdx, originalSrcLengths, isTraining);
                }
                else
                {
                    for (int i = 0; i < m_maxTgtSntSize; i++)
                    {
                        using (var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}"))
                        {
                            DecodeTransformer(tgtSnts, g, encOutput, decoder as TransformerDecoder, tgtEmbedding, posEmbedding, batchSize, deviceIdIdx, originalSrcLengths, isTraining);

                            bool allSntsEnd = true;
                            for (int j = 0; j < tgtSnts.Count; j++)
                            {
                                if (tgtSnts[j][tgtSnts[j].Count - 1] != ParallelCorpus.EOS)
                                {
                                    allSntsEnd = false;
                                    break;
                                }
                            }

                            if (allSntsEnd)
                            {
                                break;
                            }
                        }
                    }

                    RemoveDuplicatedEOS(tgtSnts);
                    return 0.0f;
                }
            }
        }





        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        /// <param name="g"></param>
        /// <param name="srcSnts"></param>
        /// <param name="encoder"></param>
        /// <param name="reversEncoder"></param>
        /// <param name="Embedding"></param>
        /// <returns></returns>
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> srcSnts, IEncoder encoder, IWeightTensor Embedding, IWeightTensor srcSelfMask, IWeightTensor posEmbedding)
        {
            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            List<IWeightTensor> inputs = new List<IWeightTensor>();

            // Generate batch-first based input embeddings
            for (int j = 0; j < batchSize; j++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    int ix_source = m_modelMetaData.Vocab.GetSourceWordIndex(srcSnts[j][i], logUnk: true);

                    var emb = g.PeekRow(Embedding, ix_source);
                    inputs.Add(emb);
                }
            }

            var inputEmbs = g.ConcatRows(inputs);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = AddPositionEmbedding(g, posEmbedding, batchSize, seqLen, inputEmbs);
            }


            return encoder.Encode(inputEmbs, batchSize, g, srcSelfMask);
        }



        private float DecodeTransformer(List<List<string>> tgtSeqs, IComputeGraph g, IWeightTensor encOutputs, TransformerDecoder decoder,
            IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, int batchSize, int deviceId, List<int> srcOriginalLenghts, bool isTraining = true)
        {
            float cost = 0.0f;

            var tgtOriginalLengths = ParallelCorpus.PadSentences(tgtSeqs);
            int tgtSeqLen = tgtSeqs[0].Count;
            int srcSeqLen = encOutputs.Rows / batchSize;

            using (IWeightTensor srcTgtMask = MaskUtils.BuildSrcTgtMask(g, srcSeqLen, tgtSeqLen, tgtOriginalLengths, srcOriginalLenghts, deviceId))
            {
                using (IWeightTensor tgtSelfTriMask = MaskUtils.BuildPadSelfTriMask(g, tgtSeqLen, tgtOriginalLengths, deviceId))
                {
                    List<IWeightTensor> inputs = new List<IWeightTensor>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        for (int j = 0; j < tgtSeqLen; j++)
                        {
                            int ix_targets_k = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSeqs[i][j], logUnk: true);
                            inputs.Add(g.PeekRow(tgtEmbedding, ix_targets_k));
                        }
                    }

                    IWeightTensor inputEmbs = inputs.Count > 1 ? g.ConcatRows(inputs) : inputs[0];

                    inputEmbs = AddPositionEmbedding(g, posEmbedding, batchSize, tgtSeqLen, inputEmbs);

                    IWeightTensor decOutput = decoder.Decode(inputEmbs, encOutputs, tgtSelfTriMask, srcTgtMask, batchSize, g);

                    //  decOutput = g.Mul(decOutput, g.Transpose(tgtEmbedding));

                    using (IWeightTensor probs = g.Softmax(decOutput, runGradients: false, inPlace: true))
                    {
                        //if (isTraining)
                        //{
                        //    var leftShiftInputSeqs = ParallelCorpus.LeftShiftSnts(tgtSeqs, ParallelCorpus.EOS);
                        //    for (int i = 0; i < batchSize; i++)
                        //    {
                        //        for (int j = 0; j < tgtSeqLen; j++)
                        //        {
                        //            using (IWeightTensor probs_i_j = g.PeekRow(probs, i * tgtSeqLen + j, runGradients: false))
                        //            {
                        //                if (j < tgtOriginalLengths[i])
                        //                {
                        //                    int ix_targets_i_j = m_modelMetaData.Vocab.GetTargetWordIndex(leftShiftInputSeqs[i][j], logUnk: true);
                        //                    float score_i_j = probs_i_j.GetWeightAt(ix_targets_i_j);

                        //                    cost += (float)-Math.Log(score_i_j);

                        //                    probs_i_j.SetWeightAt(score_i_j - 1, ix_targets_i_j);
                        //                }
                        //                else
                        //                {
                        //                    probs_i_j.CleanWeight();
                        //                }
                        //            }
                        //        }
                        //    }

                        //    decOutput.CopyWeightsToGradients(probs);
                        //}
                        if (isTraining)
                        {
                            var leftShiftInputSeqs = ParallelCorpus.LeftShiftSnts(tgtSeqs, ParallelCorpus.EOS);
                            int[] targetIds = new int[batchSize * tgtSeqLen];
                            int ids = 0;
                            for (int i = 0; i < batchSize; i++)
                            {
                                for (int j = 0; j < tgtSeqLen; j++)
                                {
                                    targetIds[ids] = j < tgtOriginalLengths[i] ? m_modelMetaData.Vocab.GetTargetWordIndex(leftShiftInputSeqs[i][j], logUnk: true) : -1;
                                    ids++;
                                }
                            }

                            cost += g.UpdateCost(probs, targetIds);
                            decOutput.CopyWeightsToGradients(probs);
                        }
                        else
                        {
                            // Output "i"th target word
                            int[] targetIdx = g.Argmax(probs, 1);
                            List<string> targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());

                            for (int i = 0; i < batchSize; i++)
                            {
                                tgtSeqs[i].Add(targetWords[i * tgtSeqLen + tgtSeqLen - 1]);
                            }
                        }
                    }
                }
            }


            return cost;
        }

        private IWeightTensor AddPositionEmbedding(IComputeGraph g, IWeightTensor posEmbedding, int batchSize, int seqLen, IWeightTensor inputEmbs)
        {
            using (var posEmbeddingPeek = g.PeekRow(posEmbedding, 0, seqLen, false))
            {
                using (var posEmbeddingPeekView = g.View(posEmbeddingPeek, runGradient: false, dims: new long[] { 1, seqLen, m_modelMetaData.EmbeddingDim }))
                {
                    using (var posEmbeddingPeekViewExp = g.Expand(posEmbeddingPeekView, runGradient: false, dims: new long[] { batchSize, seqLen, m_modelMetaData.EmbeddingDim }))
                    {
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize, seqLen, m_modelMetaData.EmbeddingDim });
                        inputEmbs = g.Add(inputEmbs, posEmbeddingPeekViewExp, true, false);
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize * seqLen, m_modelMetaData.EmbeddingDim });
                    }
                }
            }

            inputEmbs = g.Dropout(inputEmbs, batchSize, m_dropoutRatio, inPlace: true);

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
        private float DecodeAttentionLSTM(List<List<string>> outputSnts, IComputeGraph g, IWeightTensor encOutputs, AttentionDecoder decoder, IWeightTensor tgtEmbedding, int batchSize, bool isTraining = true)
        {
            float cost = 0.0f;
            int[] ix_inputs = new int[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = m_modelMetaData.Vocab.GetTargetWordIndex(outputSnts[i][0]);
            }

            // Initialize variables accoridng to current mode
            List<int> originalOutputLengths = isTraining ? ParallelCorpus.PadSentences(outputSnts) : null;
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
                    inputs.Add(g.PeekRow(tgtEmbedding, ix_inputs[j]));
                }
                IWeightTensor inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                IWeightTensor eOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);

                //Softmax for output
                using (IWeightTensor probs = g.Softmax(eOutput, runGradients: false, inPlace: true))
                {
                    //if (isTraining)
                    //{
                    //    //Calculate loss for each word in the batch
                    //    for (int k = 0; k < batchSize; k++)
                    //    {
                    //        using (IWeightTensor probs_k = g.PeekRow(probs, k, runGradients: false))
                    //        {
                    //            int ix_targets_k = m_modelMetaData.Vocab.GetTargetWordIndex(outputSnts[k][i]);
                    //            float score_k = probs_k.GetWeightAt(ix_targets_k);
                    //            if (i < originalOutputLengths[k])
                    //            {
                    //                var lcost = (float)-Math.Log(score_k);
                    //                if (float.IsNaN(lcost))
                    //                {
                    //                    Logger.WriteLine($"Score = '{score_k}' Cost = Nan at index '{i}' word '{outputSnts[k][i]}'");
                    //                    Logger.WriteLine($"Output Sentence = '{String.Join(" ", outputSnts[k])}'");
                    //                }
                    //                else
                    //                {
                    //                    cost += lcost;
                    //                }
                    //            }

                    //            probs_k.SetWeightAt(score_k - 1, ix_targets_k);
                    //            ix_inputs[k] = ix_targets_k;
                    //        }
                    //    }
                    //    eOutput.CopyWeightsToGradients(probs);
                    //}
                    if (isTraining)
                    {
                        //Calculate loss for each word in the batch
                        int[] targetIds = new int[batchSize];
                        int ids = 0;
                        for (int k = 0; k < batchSize; k++)
                        {
                            int targetsId_k = m_modelMetaData.Vocab.GetTargetWordIndex(outputSnts[k][i]);
                            targetIds[ids] = i < originalOutputLengths[k] ? targetsId_k : -1;
                            ix_inputs[k] = targetsId_k;

                            ids++;
                        }

                        cost += g.UpdateCost(probs, targetIds);
                        eOutput.CopyWeightsToGradients(probs);

                    }
                    else
                    {
                        // Output "i"th target word
                        int[] targetIdx = g.Argmax(probs, 1);
                        List<string> targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());
                        for (int j = 0; j < targetWords.Count; j++)
                        {
                            if (setEndSentId.Contains(j) == false)
                            {
                                outputSnts[j].Add(targetWords[j]);

                                if (targetWords[j] == ParallelCorpus.EOS)
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

        /// <summary>
        /// Given input sentence and generate output sentence by seq2seq model with beam search
        /// </summary>
        /// <param name="input"></param>
        /// <param name="beamSearchSize"></param>
        /// <param name="maxOutputLength"></param>
        /// <returns></returns>
        public List<List<string>> Predict(List<string> input, int beamSearchSize = 1, int maxOutputLength = 100)
        {
            (IEncoder encoder, IDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor posEmbedding) = GetNetworksOnDeviceAt(-1);
            List<List<string>> inputSeqs = ParallelCorpus.ConstructInputTokens(input);
            int batchSize = 1; // For predict with beam search, we currently only supports one sentence per call

            IComputeGraph g = CreateComputGraph(m_defaultDeviceId, needBack: false);
            AttentionDecoder rnnDecoder = decoder as AttentionDecoder;

            encoder.Reset(g.GetWeightFactory(), batchSize);
            rnnDecoder.Reset(g.GetWeightFactory(), batchSize);

            // Construct beam search status list
            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();
            BeamSearchStatus bss = new BeamSearchStatus();
            bss.OutputIds.Add((int)SENTTAGS.START);
            bss.CTs = rnnDecoder.GetCTs();
            bss.HTs = rnnDecoder.GetHTs();
            bssList.Add(bss);

            IWeightTensor encodedWeightMatrix = Encode(g, inputSeqs, encoder, srcEmbedding, null, posEmbedding);
            AttentionPreProcessResult attPreProcessResult = rnnDecoder.PreProcess(encodedWeightMatrix, batchSize, g);

            List<BeamSearchStatus> newBSSList = new List<BeamSearchStatus>();
            bool finished = false;
            int outputLength = 0;
            while (finished == false && outputLength < maxOutputLength)
            {
                finished = true;
                for (int i = 0; i < bssList.Count; i++)
                {
                    bss = bssList[i];
                    if (bss.OutputIds[bss.OutputIds.Count - 1] == (int)SENTTAGS.END)
                    {
                        newBSSList.Add(bss);
                    }
                    else if (bss.OutputIds.Count > maxOutputLength)
                    {
                        newBSSList.Add(bss);
                    }
                    else
                    {
                        finished = false;
                        int ix_input = bss.OutputIds[bss.OutputIds.Count - 1];
                        rnnDecoder.SetCTs(bss.CTs);
                        rnnDecoder.SetHTs(bss.HTs);

                        IWeightTensor x = g.PeekRow(tgtEmbedding, ix_input);
                        IWeightTensor eOutput = rnnDecoder.Decode(x, attPreProcessResult, batchSize, g);
                        using (IWeightTensor probs = g.Softmax(eOutput))
                        {
                            List<int> preds = probs.GetTopNMaxWeightIdx(beamSearchSize);
                            for (int j = 0; j < preds.Count; j++)
                            {
                                BeamSearchStatus newBSS = new BeamSearchStatus();
                                newBSS.OutputIds.AddRange(bss.OutputIds);
                                newBSS.OutputIds.Add(preds[j]);

                                newBSS.CTs = rnnDecoder.GetCTs();
                                newBSS.HTs = rnnDecoder.GetHTs();

                                float score = probs.GetWeightAt(preds[j]);
                                newBSS.Score = bss.Score;
                                newBSS.Score += (float)(-Math.Log(score));

                                //var lengthPenalty = Math.Pow((5.0f + newBSS.OutputIds.Count) / 6, 0.6);
                                //newBSS.Score /= (float)lengthPenalty;

                                newBSSList.Add(newBSS);
                            }
                        }
                    }
                }

                bssList = BeamSearch.GetTopNBSS(newBSSList, beamSearchSize);
                newBSSList.Clear();

                outputLength++;
            }

            // Convert output target word ids to real string
            List<List<string>> results = new List<List<string>>();
            for (int i = 0; i < bssList.Count; i++)
            {
                results.Add(m_modelMetaData.Vocab.ConvertTargetIdsToString(bssList[i].OutputIds));
            }

            return results;
        }

        //public void VisualizeNeuralNetwork(string visNNFilePath)
        //{
        //    (IEncoder encoder, IDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding) = GetNetworksOnDeviceAt(-1);
        //    // Build input sentence
        //    List<List<string>> inputSeqs = ParallelCorpus.ConstructInputTokens(null);
        //    int batchSize = inputSeqs.Count;
        //    IComputeGraph g = CreateComputGraph(m_defaultDeviceId, needBack: false, visNetwork: true);
        //    AttentionDecoder rnnDecoder = decoder as AttentionDecoder;

        //    encoder.Reset(g.GetWeightFactory(), batchSize);
        //    rnnDecoder.Reset(g.GetWeightFactory(), batchSize);

        //    // Run encoder
        //    IWeightTensor encodedWeightMatrix = Encode(g, inputSeqs, encoder, srcEmbedding, null);

        //    // Prepare for attention over encoder-decoder
        //    AttentionPreProcessResult attPreProcessResult = rnnDecoder.PreProcess(encodedWeightMatrix, batchSize, g);

        //    // Run decoder
        //    IWeightTensor x = g.PeekRow(tgtEmbedding, (int)SENTTAGS.START);
        //    IWeightTensor eOutput = rnnDecoder.Decode(x, attPreProcessResult, batchSize, g);
        //    IWeightTensor probs = g.Softmax(eOutput);

        //    g.VisualizeNeuralNetToFile(visNNFilePath);
        //}

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_modelMetaData.Vocab.DumpSourceVocab(outputSrcVocab);
            m_modelMetaData.Vocab.DumpTargetVocab(outputTgtVocab);
        }
    }
}
