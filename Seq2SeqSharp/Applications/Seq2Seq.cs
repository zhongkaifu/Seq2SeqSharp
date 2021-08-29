

using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{
    public class Seq2Seq : BaseSeq2SeqFramework
    {
        private readonly IModel m_modelMetaData;

        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IWeightTensor> m_sharedEmbedding; //The embeddings over devices for both source and target

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after all layers in decoder

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        readonly Seq2SeqOptions m_options = null;

        public Seq2Seq(Seq2SeqOptions options, Vocab srcVocab = null, Vocab tgtVocab = null)
            : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || tgtVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                // Model file exists, so we load it from file.
                m_modelMetaData = LoadModel(CreateTrainableParameters);
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);
                DecoderTypeEnums decoderType = (DecoderTypeEnums)Enum.Parse(typeof(DecoderTypeEnums), options.DecoderType);

                m_modelMetaData = new Seq2SeqModel(options.HiddenSize, options.SrcEmbeddingDim, options.TgtEmbeddingDim, options.EncoderLayerDepth, options.DecoderLayerDepth, options.MultiHeadNum,
                    encoderType, decoderType, srcVocab, tgtVocab, options.EnableCoverageModel, options.SharedEmbeddings, options.EnableSegmentEmbeddings, options.ApplyContextEmbeddingsToEntireSequence);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        private bool CreateTrainableParameters(IModel modelMetaData)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(modelMetaData, m_options, raDeviceIds);
            m_decoder = Decoder.CreateDecoders(modelMetaData, m_options, raDeviceIds, contextDim);

            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", modelMetaData.HiddenDim, modelMetaData.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true, learningRateFactor: m_options.DecoderStartLearningRateFactor), DeviceIds);


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
                m_sharedEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SharedEmbeddings", isTrainable: m_options.IsSrcEmbeddingTrainable, learningRateFactor: m_options.EncoderStartLearningRateFactor), DeviceIds);

                m_srcEmbedding = null;
                m_tgtEmbedding = null;
            }
            else
            {
                Logger.WriteLine($"Creating embeddings for source side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
                m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsSrcEmbeddingTrainable, learningRateFactor: m_options.EncoderStartLearningRateFactor), DeviceIds);

                Logger.WriteLine($"Creating embeddings for target side. Shape = '({modelMetaData.TgtVocab.Count} ,{modelMetaData.DecoderEmbeddingDim})'");
                m_tgtEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.TgtVocab.Count, modelMetaData.DecoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: m_options.IsTgtEmbeddingTrainable, learningRateFactor: m_options.DecoderStartLearningRateFactor), DeviceIds);

                m_sharedEmbedding = null;
            }

            return true;
        }



        public void Train(int maxTrainingEpoch, Seq2SeqCorpus trainCorpus, List<Seq2SeqCorpus> validCorpusList, ILearningRate learningRate, List<IMetric> metrics, IOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>
            {
                { 0, metrics }
            };

            Dictionary<string, IEnumerable<ISntPairBatch>> validCorpusDict = new Dictionary<string, IEnumerable<ISntPairBatch>>();
            string primaryValidCorpusName = "";
            if (validCorpusList != null && validCorpusList.Count > 0)
            {
                primaryValidCorpusName = validCorpusList[0].CorpusName;
                foreach (var item in validCorpusList)
                {
                    validCorpusDict.Add(item.CorpusName, item);
                }
            }

            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpusDict, primaryValidCorpusName, learningRate, optimizer, taskId2metrics, m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        public void Valid(Seq2SeqCorpus validCorpus, List<IMetric> metrics)
        {
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>
            {
                { 0, metrics }
            };
            RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, true);
        }

        public NetworkResult Test(List<List<List<string>>> inputTokensGroups)
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch();
            spb.CreateBatch(inputTokensGroups);

            var nrs = RunTest(spb, RunForwardOnSingleDevice);

            return nrs[0];
        }


        public void Test()
        {
            SntPairBatchStreamReader<Seq2SeqCorpusBatch> reader = new SntPairBatchStreamReader<Seq2SeqCorpusBatch>(m_options.InputTestFile, m_options.BatchSize, m_options.MaxTestSrcSentLength);
            SntPairBatchStreamWriter writer = new SntPairBatchStreamWriter(m_options.OutputFile);
            RunTest<Seq2SeqCorpusBatch>(reader, writer, RunForwardOnSingleDevice);
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IDecoder, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), 
                    m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_sharedEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx), 
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx));
        }


        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, int deviceIdIdx, bool isTraining)
        {
            (IEncoder encoder, IDecoder decoder, IFeedForwardLayer decoderFFLayer, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);

            if (isTraining && srcSnts[0].Count > m_options.MaxTrainSrcSentLength + 2)
            {
                throw new InvalidDataException($"The source sentence is too long. Its length = '{srcSnts[0].Count}', but MaxTrainSrcSentLength is '{m_options.MaxTrainSrcSentLength}'. The sentence is '{String.Join(" ", srcSnts[0])}'");
            }

            IWeightTensor encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, segmentEmbedding, srcSnts, originalSrcLengths);

            List<NetworkResult> nrs = new List<NetworkResult>();

            // Generate output decoder sentences
            int batchSize = srcSnts.Count;
            var tgtSnts = sntPairBatch.GetTgtTokens(0);
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);
            NetworkResult nr = new NetworkResult();

            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            if (decoder is AttentionDecoder)
            {
                nr.Cost = Decoder.DecodeAttentionLSTM(tgtTokensList, computeGraph, encOutput, decoder as AttentionDecoder, decoderFFLayer, tgtEmbedding, m_modelMetaData.TgtVocab, srcSnts.Count, isTraining);
                nr.Output = new List<List<List<string>>>
                {
                    m_modelMetaData.TgtVocab.ConvertIdsToString(tgtTokensList)
                };
            }
            else
            {
                if (isTraining)
                {
                    (var c, _) = Decoder.DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding, originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType, m_options.DropoutRatio, isTraining);
                    nr.Cost = c;
                    nr.Output = null;
                }
                else
                {
                    List<List<BeamSearchStatus>> beam2batchStatus = Decoder.InitBeamSearchStatusListList(batchSize, tgtTokensList);
                    for (int i = 0; i < m_options.MaxTestTgtSentLength; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = null; //(batch_size, beam_search_size)
                        try
                        {
                            foreach (var batchStatus in beam2batchStatus)
                            {
                                var batch2tgtTokens = Decoder.ExtractBatchTokens(batchStatus);
                                using var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}");
                                (var cost2, var bssSeqList) = Decoder.DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                                                originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType, 0.0f, isTraining, beamSearchSize: m_options.BeamSearchSize,
                                                                                outputSentScore: m_options.BeamSearchSize > 1, previousBeamSearchResults: batchStatus);

                                bssSeqList = Decoder.SwapBeamAndBatch(bssSeqList);
                                batch2beam2seq = Decoder.MergeTwoBeamSearchStatus(batch2beam2seq, bssSeqList);
                            }
                        }
                        catch (OutOfMemoryException)
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


                        beam2batchStatus = Decoder.SwapBeamAndBatch(batch2beam2seq);
                        if (Decoder.AreAllSentsCompleted(beam2batchStatus))
                        {
                            break;
                        }                        
                    }

                    nr.Cost = 0.0f;
                    nr.Output = m_modelMetaData.TgtVocab.ExtractTokens(beam2batchStatus);
                }
            }

            nr.RemoveDuplicatedEOS();

            nrs.Add(nr);
            return nrs;
        }



        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_modelMetaData.SrcVocab.DumpVocab(outputSrcVocab);
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
        }
    }
}
