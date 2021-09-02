using AdvUtils;
using Microsoft.Extensions.Caching.Memory;
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
    public class SeqSimilarity : BaseSeq2SeqFramework
    {
        private readonly IModel m_modelMetaData;
        public Vocab SrcVocab => m_modelMetaData.SrcVocab;
        public Vocab ClsVocab => m_modelMetaData.ClsVocab;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        readonly SeqSimilarityOptions m_options = null;


        private MemoryCache m_memoryCache;

        public SeqSimilarity(SeqSimilarityOptions options, Vocab srcVocab = null, Vocab clsVocab = null)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;

            m_memoryCache = new MemoryCache(new MemoryCacheOptions
            {
                SizeLimit = 1024
            });

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || clsVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                m_modelMetaData = LoadModel(CreateTrainableParameters);
            }
            else
            {
                EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);

                m_modelMetaData = new SeqSimilarityModel(options.HiddenSize, options.EmbeddingDim, options.EncoderLayerDepth, options.MultiHeadNum,
                    encoderType, srcVocab, clsVocab, options.EnableSegmentEmbeddings, m_options.SimilarityType);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }


        public void Train(int maxTrainingEpoch, SeqClassificationMultiTasksCorpus trainCorpus, List<SeqClassificationMultiTasksCorpus> validCorpusList, ILearningRate learningRate, IMetric metric, IOptimizer optimizer)
        {
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
            taskId2metrics.Add(0, new List<IMetric>());
            taskId2metrics[0].Add(metric);

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

            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpusDict, primaryValidCorpusName, learningRate, optimizer, taskId2metrics, m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        public void Valid(SeqClassificationMultiTasksCorpus validCorpus, Dictionary<int, List<IMetric>> taskId2metrics)
        {
            RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, true);
        }

        public List<NetworkResult> Test(List<List<List<string>>> inputTokens)
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch();
            spb.CreateBatch(inputTokens);

            return RunTest(spb, RunForwardOnSingleDevice);
        }


        public void Test()
        {
            SntPairBatchStreamReader<SeqClassificationMultiTasksCorpusBatch> reader = new SntPairBatchStreamReader<SeqClassificationMultiTasksCorpusBatch>(m_options.InputTestFile, m_options.BatchSize, m_options.MaxTestSentLength);
            SntPairBatchStreamWriter writer = new SntPairBatchStreamWriter(m_options.OutputFile);
            RunTest<SeqClassificationMultiTasksCorpusBatch>(reader, writer, RunForwardOnSingleDevice);
        }



        private bool CreateTrainableParameters(IModel modelMetaData)
        {
            Logger.WriteLine($"Creating encoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(modelMetaData, m_options, raDeviceIds);

            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer($"FeedForward_Encoder", contextDim, modelMetaData.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                    Math.Max(m_options.MaxTrainSentLength, m_options.MaxTestSentLength) + 2,
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

            Logger.WriteLine($"Creating embeddings. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsEmbeddingTrainable), DeviceIds);

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, IFeedForwardLayer, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_encoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx));
        }


        private string GenerateCacheKey(List<List<string>> strs)
        {
            List<string> r = new List<string>();

            foreach (var str in strs)
            {
                r.Add(String.Join(" ", str));
            }

            return String.Join("\t", r);
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
            int batchSize = sntPairBatch.BatchSize;

            List<NetworkResult> nrs = new List<NetworkResult>();
            float cost = 0.0f;
            NetworkResult nr = new NetworkResult
            {
                Output = new List<List<List<string>>>()
            };

            (IEncoder encoder, IWeightTensor srcEmbedding, IFeedForwardLayer encoderFFLayer, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            IWeightTensor encOutput1 = null;
            IWeightTensor encOutput2 = null;
            if (isTraining == false && m_options.ProcessorType.Equals("CPU", StringComparison.InvariantCultureIgnoreCase))
            {
                //We only check cache at inference time
                string cacheKey1 = GenerateCacheKey(sntPairBatch.GetSrcTokens(0));
                if (!m_memoryCache.TryGetValue(cacheKey1, out encOutput1))
                {
                    encOutput1 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, m_shuffleType, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 0); // output shape: [batch_size, dim]

                    var cacheEntryOptions = new MemoryCacheEntryOptions().SetSize(1);
                    m_memoryCache.Set(cacheKey1, encOutput1, cacheEntryOptions);
                }

                string cacheKey2 = GenerateCacheKey(sntPairBatch.GetSrcTokens(1));
                if (!m_memoryCache.TryGetValue(cacheKey2, out encOutput2))
                {
                    encOutput2 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, m_shuffleType, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 1); // output_shape: [batch_size, dim]

                    var cacheEntryOptions = new MemoryCacheEntryOptions().SetSize(1);
                    m_memoryCache.Set(cacheKey2, encOutput2, cacheEntryOptions);
                }
            }
            else
            {
                //We always run encoder network during training time or using GPUs
                encOutput1 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, m_shuffleType, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 0); // output shape: [batch_size, dim]
                encOutput2 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, m_shuffleType, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 1); // output_shape: [batch_size, dim]
            }

            if (m_modelMetaData.SimilarityType.Equals("Continuous", StringComparison.InvariantCultureIgnoreCase))
            {
                // Cosine similairy
                var w12 = computeGraph.EltMul(encOutput1, encOutput2);
                w12 = computeGraph.Sum(w12, 1);
                var w1 = computeGraph.EltMul(encOutput1, encOutput1);
                w1 = computeGraph.Sum(w1, 1);
                var w2 = computeGraph.EltMul(encOutput2, encOutput2);
                w2 = computeGraph.Sum(w2, 1);
                var n12 = computeGraph.EltMul(w1, w2);
                n12 = computeGraph.Rsqrt(n12);
                var probs = computeGraph.EltMul(w12, n12);
                if (isTraining)
                {
                    var tgtSnts = sntPairBatch.GetTgtTokens(0);
                    for (int k = 0; k < batchSize; k++)
                    {
                        float golden_score_k = float.Parse(tgtSnts[k][0]); // Get golden similiary score from target side
                        float score_k = probs.GetWeightAt(new long[] { k, 0 });

                        probs.SetWeightAt(score_k - golden_score_k, new long[] { k, 0 });
                        cost += (float)Math.Abs(score_k - golden_score_k);
                    }

                    probs.CopyWeightsToGradients(probs);
                    nr.Cost = cost / batchSize;
                }
                else
                {
                    nr.Output.Add(new List<List<string>>());
                    for (int k = 0; k < batchSize; k++)
                    {
                        float score_k = probs.GetWeightAt(new long[] { k, 0 });

                        nr.Output[0].Add(new List<string>());
                        nr.Output[0][k].Add(score_k.ToString());
                    }
                }
            }
            else
            {
                IWeightTensor encOutput = computeGraph.EltMul(encOutput1, encOutput2);
                IWeightTensor ffLayer = encoderFFLayer.Process(encOutput, batchSize, computeGraph);
                using (IWeightTensor probs = computeGraph.Softmax(ffLayer, runGradients: false, inPlace: true))
                {
                    if (isTraining)
                    {
                        var tgtSnts = sntPairBatch.GetTgtTokens(0);
                        for (int k = 0; k < batchSize; k++)
                        {
                            int ix_targets_k_j = m_modelMetaData.ClsVocab.GetWordIndex(tgtSnts[k][0]);
                            float score_k = probs.GetWeightAt(new long[] { k, ix_targets_k_j });
                            cost += (float)-Math.Log(score_k);
                            probs.SetWeightAt(score_k - 1, new long[] { k, ix_targets_k_j });
                        }

                        ffLayer.CopyWeightsToGradients(probs);

                        nr.Cost = cost / batchSize;
                    }
                    else
                    {
                        // Output "i"th target word
                        using var targetIdxTensor = computeGraph.Argmax(probs, 1);
                        float[] targetIdx = targetIdxTensor.ToWeightArray();
                        List<string> targetWords = m_modelMetaData.ClsVocab.ConvertIdsToString(targetIdx.ToList());
                        nr.Output.Add(new List<List<string>>());

                        for (int k = 0; k < batchSize; k++)
                        {
                            nr.Output[0].Add(new List<string>());
                            nr.Output[0][k].Add(targetWords[k]);
                        }
                    }
                }
            }


            nrs.Add(nr);

            return nrs;
        }
    }
}
