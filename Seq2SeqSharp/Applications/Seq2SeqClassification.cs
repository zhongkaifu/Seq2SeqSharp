using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class Seq2SeqClassification : BaseSeq2SeqFramework
    {
        private readonly IModel m_modelMetaData;

        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IWeightTensor> m_sharedEmbedding; //The embeddings over devices for both source and target

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after all layers in decoder

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;

        Seq2SeqClassificationOptions m_options = null;

        public Vocab ClsVocab => m_modelMetaData.ClsVocab;


        public Seq2SeqClassification(Seq2SeqClassificationOptions options, Vocab srcVocab = null, Vocab tgtVocab = null, Vocab clsVocab = null)
            : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours, options.PrimaryTaskId)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || tgtVocab != null || clsVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                m_modelMetaData = LoadModel(CreateTrainableParameters);
            }
            else
            {
                EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);
                DecoderTypeEnums decoderType = (DecoderTypeEnums)Enum.Parse(typeof(DecoderTypeEnums), options.DecoderType);

                m_modelMetaData = new Seq2SeqClassificationModel(options.HiddenSize, options.SrcEmbeddingDim, options.TgtEmbeddingDim, options.EncoderLayerDepth, options.DecoderLayerDepth, options.MultiHeadNum,
                    encoderType, decoderType, srcVocab, tgtVocab, clsVocab, options.EnableCoverageModel, options.SharedEmbeddings, options.EnableSegmentEmbeddings);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            string primaryTaskStr = options.PrimaryTaskId == 0 ? "Sequence Generation" : "Sequence Classification";

            Logger.WriteLine($"Encoder is trainable: '{options.IsEncoderTrainable}'");
            Logger.WriteLine($"Decoder is trainable: '{options.IsDecoderTrainable}'");
            Logger.WriteLine($"Max source sentence length in training corpus = '{options.MaxTrainSrcSentLength}'");
            Logger.WriteLine($"Max target sentence length in training corpus = '{options.MaxTrainTgtSentLength}'");
            Logger.WriteLine($"BeamSearch Size = '{options.BeamSearchSize}'");
            Logger.WriteLine($"Shared embeddings = '{options.SharedEmbeddings}'");
            Logger.WriteLine($"Enable segment embeddings = '{options.EnableSegmentEmbeddings}'");
            Logger.WriteLine($"Primary task = '{primaryTaskStr}'");
        }

        private bool CreateTrainableParameters(IModel modelMetaData)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim = 0;
            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: m_options.IsEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, m_options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: m_options.IsEncoderTrainable), DeviceIds);

                contextDim = modelMetaData.HiddenDim;
            }

            if (modelMetaData.DecoderType == DecoderTypeEnums.AttentionLSTM)
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                     new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.DecoderEmbeddingDim, contextDim,
                     m_options.DropoutRatio, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem(), modelMetaData.EnableCoverageModel, isTrainable: m_options.IsDecoderTrainable), DeviceIds);
            }
            else
            {
                m_decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new TransformerDecoder("TransformerDecoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.DecoderEmbeddingDim, modelMetaData.DecoderLayerDepth, m_options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: m_options.IsDecoderTrainable), DeviceIds);
            }

            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Encoder_0", modelMetaData.HiddenDim, modelMetaData.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true), DeviceIds);

            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", modelMetaData.HiddenDim, modelMetaData.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true), DeviceIds);


            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer || modelMetaData.DecoderType == DecoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                    Math.Max(Math.Max(m_options.MaxTrainSrcSentLength, m_options.MaxTestSrcSentLength), Math.Max(m_options.MaxTrainTgtSentLength, m_options.MaxTestTgtSentLength)) + 2,
                    contextDim, DeviceIds[0], "PosEmbedding", false), DeviceIds, true);

                if (modelMetaData.EnableSegmentEmbeddings)
                {
                    m_segmentEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { 16, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SegmentEmbedding", isTrainable: true), DeviceIds);
                }
                else
                {
                    m_segmentEmbedding = null;
                }
            }
            else
            {
                m_posEmbedding = null;
                m_segmentEmbedding = null;
            }

            if (modelMetaData.SharedEmbeddings)
            {
                Logger.WriteLine($"Creating shared embeddings for both source side and target side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
                m_sharedEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SharedEmbeddings", isTrainable: m_options.IsSrcEmbeddingTrainable), DeviceIds);

                m_srcEmbedding = null;
                m_tgtEmbedding = null;
            }
            else
            {
                Logger.WriteLine($"Creating embeddings for source side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
                m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsSrcEmbeddingTrainable), DeviceIds);

                Logger.WriteLine($"Creating embeddings for target side. Shape = '({modelMetaData.TgtVocab.Count} ,{modelMetaData.DecoderEmbeddingDim})'");
                m_tgtEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.TgtVocab.Count, modelMetaData.DecoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: m_options.IsTgtEmbeddingTrainable), DeviceIds);

                m_sharedEmbedding = null;
            }

            return true;
        }

        public void Train(int maxTrainingEpoch, Seq2SeqClassificationCorpus trainCorpus, Seq2SeqClassificationCorpus validCorpus, ILearningRate learningRate, Dictionary<int, List<IMetric>> taskId2metrics, IOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, taskId2metrics, m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        public void Valid(Seq2SeqClassificationCorpus validCorpus, List<IMetric> metrics)
        {
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
            taskId2metrics.Add(0, metrics);
            RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, true);
        }

        public List<NetworkResult> Test(List<List<string>> inputTokens, int beamSearchSize)
        {
            Seq2SeqClassificationCorpusBatch spb = new Seq2SeqClassificationCorpusBatch();
            spb.CreateBatch(inputTokens);

            var nrs = RunTest(spb, beamSearchSize, RunForwardOnSingleDevice);

            return nrs;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IDecoder, IFeedForwardLayer, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_encoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding == null ? null : m_posEmbedding.GetNetworkOnDevice(deviceIdIdx), m_segmentEmbedding == null ? null : m_segmentEmbedding.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, int deviceIdIdx, bool isTraining)
        {
            List<NetworkResult> nrs = new List<NetworkResult>();

            var srcSnts = sntPairBatch.GetSrcTokens(0);

            (IEncoder encoder, IDecoder decoder, IFeedForwardLayer encoderFFLayer, IFeedForwardLayer decoderFFLayer, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);
            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            List<int> originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            int srcSeqPaddedLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;
            IWeightTensor srcSelfMask = m_shuffleType == ShuffleEnums.NoPaddingInSrc ? null : computeGraph.BuildPadSelfMask(srcSeqPaddedLen, originalSrcLengths); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);
            IWeightTensor encOutput = Encode(computeGraph, srcTokensList, encoder, srcEmbedding, srcSelfMask, posEmbedding, originalSrcLengths, segmentEmbedding);

            if (srcSelfMask != null)
            {
                srcSelfMask.Dispose();
            }


            float[] clsIdxs = new float[batchSize];

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < srcSnts[i].Count; j++)
                {
                    if (srcSnts[i][j] == BuildInTokens.CLS)
                    {
                        clsIdxs[i] = i * srcSeqPaddedLen + j;
                        break;
                    }
                }
            }

            IWeightTensor clsWeightTensor = computeGraph.IndexSelect(encOutput, clsIdxs);

            float cost = 0.0f;
            NetworkResult nrCLS = new NetworkResult();
            nrCLS.Output = new List<List<List<string>>>();

            IWeightTensor ffLayer = encoderFFLayer.Process(clsWeightTensor, batchSize, computeGraph);
            using (IWeightTensor probs = computeGraph.Softmax(ffLayer, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    var clsSnts = sntPairBatch.GetTgtTokens(1);
                    for (int k = 0; k < batchSize; k++)
                    {
                        int ix_targets_k_j = m_modelMetaData.ClsVocab.GetWordIndex(clsSnts[k][0]);
                        float score_k = probs.GetWeightAt(new long[] { k, ix_targets_k_j });
                        cost += (float)-Math.Log(score_k);
                        probs.SetWeightAt(score_k - 1, new long[] { k, ix_targets_k_j });
                    }

                    ffLayer.CopyWeightsToGradients(probs);

                    nrCLS.Cost = cost / batchSize;
                }
                else
                {
                    // Output "i"th target word
                    using (var targetIdxTensor = computeGraph.Argmax(probs, 1))
                    {
                        float[] targetIdx = targetIdxTensor.ToWeightArray();
                        List<string> targetWords = m_modelMetaData.ClsVocab.ConvertIdsToString(targetIdx.ToList());
                        nrCLS.Output.Add(new List<List<string>>());

                        for (int k = 0; k < batchSize; k++)
                        {
                            nrCLS.Output[0].Add(new List<string>());
                            nrCLS.Output[0][k].Add(targetWords[k]);
                        }
                    }
                }
            }

            // Generate output decoder sentences
            var tgtSnts = sntPairBatch.GetTgtTokens(0);
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);

            NetworkResult nr = new NetworkResult();
            if (decoder is AttentionDecoder)
            {
                nr.Cost = DecodeAttentionLSTM(tgtTokensList, computeGraph, encOutput, decoder as AttentionDecoder, decoderFFLayer, tgtEmbedding, srcSnts.Count, isTraining);
                nr.Output = new List<List<List<string>>>();
                nr.Output.Add(m_modelMetaData.TgtVocab.ConvertIdsToString(tgtTokensList));
            }
            else
            {
                if (isTraining)
                {
                    (var c, var tmp) = DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                         m_shuffleType == ShuffleEnums.NoPaddingInSrc ? null : originalSrcLengths, isTraining);
                    nr.Cost = c;
                    nr.Output = null;
                }
                else
                {
                    List<List<List<int>>> beam2batch2tgtTokens = new List<List<List<int>>>(); // (beam_search_size, batch_size, tgt_token_size)
                    List<List<List<Alignment>>> beam2batch2alignment = null; // (beam_search_size, batch_size, tgt_token_size)

                    beam2batch2tgtTokens.Add(tgtTokensList);
                    for (int i = 0; i < m_options.MaxTestTgtSentLength; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = new List<List<BeamSearchStatus>>(); //(batch_size, beam_search_size)
                        for (int j = 0; j < batchSize; j++)
                        {
                            batch2beam2seq.Add(new List<BeamSearchStatus>());
                        }

                        try
                        {
                            foreach (var batch2tgtTokens in beam2batch2tgtTokens)
                            {
                                using (var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}"))
                                {
                                    (var cost2, var bssSeqList) = DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                                                    m_shuffleType == ShuffleEnums.NoPaddingInSrc ? null : originalSrcLengths, isTraining, beamSearchSize: m_options.BeamSearchSize,
                                                                                    outputAlignmentSrcPos: m_options.OutputAlignment, outputSentScore: m_options.BeamSearchSize > 1);

                                    for (int j = 0; j < m_options.BeamSearchSize; j++)
                                    {
                                        for (int k = 0; k < batchSize; k++)
                                        {
                                            batch2beam2seq[k].Add(bssSeqList[j][k]);
                                        }
                                    }
                                }
                            }
                        }
                        catch (OutOfMemoryException err)
                        {
                            Logger.WriteLine(Logger.Level.warn, $"We have out of memory while generating '{i}th' tokens, so terminate decoding for current sequences.");
                            break;
                        }

                        if (m_options.BeamSearchSize > 1)
                        {
                            // Keep top N result and drop all others
                            for (int k = 0; k < batchSize; k++)
                            {
                                batch2beam2seq[k] = BeamSearch.GetTopNBSS(batch2beam2seq[k], m_options.BeamSearchSize);
                            }
                        }

                        beam2batch2tgtTokens.Clear();
                        beam2batch2alignment = new List<List<List<Alignment>>>();
                        for (int k = 0; k < m_options.BeamSearchSize; k++)
                        {
                            beam2batch2tgtTokens.Add(new List<List<int>>());
                            beam2batch2alignment.Add(new List<List<Alignment>>());
                        }

                        // Convert shape from (batch, beam, seq) to (beam, batch, seq), and check if all output sentences are ended. If so, we will stop decoding.
                        bool allSntsEnd = true;
                        for (int j = 0; j < batchSize; j++)
                        {
                            for (int k = 0; k < m_options.BeamSearchSize; k++)
                            {
                                beam2batch2tgtTokens[k].Add(batch2beam2seq[j][k].OutputIds);
                                beam2batch2alignment[k].Add(batch2beam2seq[j][k].AlignmentToSrc);

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
                    nr.Output = m_modelMetaData.TgtVocab.ConvertIdsToString(beam2batch2tgtTokens);
                    nr.Alignment = beam2batch2alignment;
                }
            }

            nr.RemoveDuplicatedEOS();

            nrs.Add(nr);
            nrs.Add(nrCLS);

            return nrs;
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
        private IWeightTensor Encode(IComputeGraph g, List<List<int>> seqs, IEncoder encoder, IWeightTensor embeddings, IWeightTensor selfMask, IWeightTensor posEmbeddings, List<int> seqOriginalLengths, IWeightTensor segmentEmbeddings)
        {
            int batchSize = seqs.Count;
            var inputEmbs = TensorUtils.ExtractTokensEmbeddings(seqs, g, embeddings, seqOriginalLengths, segmentEmbeddings, m_modelMetaData.SrcVocab);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, m_options.DropoutRatio);
            }

            return encoder.Encode(inputEmbs, batchSize, g, selfMask);
        }



        private (float, List<List<BeamSearchStatus>>) DecodeTransformer(List<List<int>> tgtSeqs, IComputeGraph g, IWeightTensor encOutputs, TransformerDecoder decoder, IFeedForwardLayer decoderFFLayer,
            IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, List<int> srcOriginalLenghts, bool isTraining = true, int beamSearchSize = 1, bool outputAlignmentSrcPos = false, bool outputSentScore = true)
        {
            int eosTokenId = m_modelMetaData.TgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            float cost = 0.0f;

            int batchSize = tgtSeqs.Count;
            var tgtOriginalLengths = BuildInTokens.PadSentences(tgtSeqs, eosTokenId);
            int tgtSeqLen = tgtSeqs[0].Count;
            int srcSeqLen = encOutputs.Rows / batchSize;

            IWeightTensor decOutput = null;
            IWeightTensor decEncAttnProbs = null;
            using (IWeightTensor srcTgtMask = g.BuildSrcTgtMask(srcSeqLen, tgtSeqLen, tgtOriginalLengths, srcOriginalLenghts))
            {
                using (IWeightTensor tgtSelfTriMask = g.BuildPadSelfTriMask(tgtSeqLen, tgtOriginalLengths))
                {
                    IWeightTensor inputEmbs = TensorUtils.ExtractTokensEmbeddings(tgtSeqs, g, tgtEmbedding, tgtOriginalLengths, null, m_modelMetaData.TgtVocab);
                    inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbedding, batchSize, inputEmbs, m_options.DropoutRatio);
                    (decOutput, decEncAttnProbs) = decoder.Decode(inputEmbs, encOutputs, tgtSelfTriMask, srcTgtMask, batchSize, g, outputAttnWeights: outputAlignmentSrcPos);
                }
            }

            IWeightTensor ffLayer = decoderFFLayer.Process(decOutput, batchSize, g);
            IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true);

            if (isTraining)
            {
                var leftShiftTgtSeqs = g.LeftShiftTokens(tgtSeqs, eosTokenId);
                var scatterIdxTensor = g.View(leftShiftTgtSeqs, false, new long[] { tgtSeqs.Count * tgtSeqs[0].Count, 1 });
                var gatherTensor = g.Gather(probs, scatterIdxTensor, 1);

                var rnd = new TensorSharp.RandomGenerator();
                int idx = rnd.NextSeed() % (batchSize * tgtSeqLen);
                float score = gatherTensor.GetWeightAt(new long[] { idx, 0 });
                cost += (float)-Math.Log(score);

                var lossTensor = g.Add(gatherTensor, -1.0f, false);
                TensorUtils.Scatter(probs, lossTensor, scatterIdxTensor, 1);

                ffLayer.CopyWeightsToGradients(probs);

                return (cost, null);
            }
            else
            {
                // Transformer decoder with beam search at inference time
                List<List<BeamSearchStatus>> bssSeqList = new List<List<BeamSearchStatus>>();
                while (beamSearchSize > 0)
                {
                    // Output "i"th target word
                    using (var targetIdxTensor = g.Argmax(probs, 1))
                    {
                        IWeightTensor gatherTensor = null;
                        if (outputSentScore)
                        {
                            gatherTensor = g.Gather(probs, targetIdxTensor, 1);
                            gatherTensor = g.Log(gatherTensor);
                            gatherTensor = g.View(gatherTensor, false, new long[] { batchSize, tgtSeqLen, 1 });
                            gatherTensor = g.Sum(gatherTensor, 1, runGradient: false);
                        }

                        float[] targetIdx = targetIdxTensor.ToWeightArray();
                        float[] alignmentSrcPos = null;
                        float[] alignmentSrcScore = null;
                        if (outputAlignmentSrcPos)
                        {
                            using (var sourcePosIdxTensor = g.Argmax(decEncAttnProbs, 1))
                            {
                                alignmentSrcPos = sourcePosIdxTensor.ToWeightArray();
                            }

                            var sourceScoreTensor = g.Max(decEncAttnProbs, 1);
                            alignmentSrcScore = sourceScoreTensor.ToWeightArray();
                        }


                        List<BeamSearchStatus> outputTgtSeqs = new List<BeamSearchStatus>();
                        for (int i = 0; i < batchSize; i++)
                        {
                            BeamSearchStatus bss = new BeamSearchStatus();
                            bss.OutputIds.AddRange(tgtSeqs[i]);
                            bss.OutputIds.Add((int)(targetIdx[i * tgtSeqLen + tgtSeqLen - 1]));

                            if (outputSentScore)
                            {
                                bss.Score = -1.0f * gatherTensor.GetWeightAt(new long[] { i, 0, 0 });
                            }

                            if (outputAlignmentSrcPos)
                            {
                                Alignment align = new Alignment(0, 0);
                                bss.AlignmentToSrc.Add(align);
                                for (int j = 0; j < tgtSeqLen; j++)
                                {
                                    align = new Alignment((int)(alignmentSrcPos[i * tgtSeqLen + j]), alignmentSrcScore[i * tgtSeqLen + j]);
                                    bss.AlignmentToSrc.Add(align);
                                }
                            }

                            outputTgtSeqs.Add(bss);
                        }

                        bssSeqList.Add(outputTgtSeqs);

                        beamSearchSize--;
                        if (beamSearchSize > 0)
                        {
                            TensorUtils.ScatterFill(probs, 0.0f, targetIdxTensor, 1);
                        }
                    }
                }
                return (0.0f, bssSeqList);
            }
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
        private float DecodeAttentionLSTM(List<List<int>> outputSnts, IComputeGraph g, IWeightTensor encOutputs, AttentionDecoder decoder, IFeedForwardLayer decoderFFLayer, IWeightTensor tgtEmbedding, int batchSize, bool isTraining = true)
        {
            int eosTokenId = m_modelMetaData.TgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            float cost = 0.0f;
            float[] ix_inputs = new float[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = outputSnts[i][0];
            }

            // Initialize variables accoridng to current mode
            List<int> originalOutputLengths = isTraining ? BuildInTokens.PadSentences(outputSnts, eosTokenId) : null;
            int seqLen = isTraining ? outputSnts[0].Count : 64;
            HashSet<int> setEndSentId = isTraining ? null : new HashSet<int>();

            // Pre-process for attention model
            AttentionPreProcessResult attPreProcessResult = decoder.PreProcess(encOutputs, batchSize, g);
            for (int i = 1; i < seqLen; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    inputs.Add(g.Peek(tgtEmbedding, 0, (int)ix_inputs[j]));
                }
                IWeightTensor inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                IWeightTensor dOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);
                IWeightTensor ffLayer = decoderFFLayer.Process(dOutput, batchSize, g);

                //Softmax for output
                using (IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true))
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
                        ffLayer.CopyWeightsToGradients(probs);
                    }
                    else
                    {
                        // Output "i"th target word
                        using (var targetIdxTensor = g.Argmax(probs, 1))
                        {
                            float[] targetIdx = targetIdxTensor.ToWeightArray();
                            for (int j = 0; j < targetIdx.Length; j++)
                            {
                                if (setEndSentId.Contains(j) == false)
                                {
                                    outputSnts[j].Add((int)targetIdx[j]);

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
            }

            return cost;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab, string outputClsVocab)
        {
            m_modelMetaData.SrcVocab.DumpVocab(outputSrcVocab);
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
            m_modelMetaData.ClsVocab.DumpVocab(outputClsVocab);
        }
    }
}
