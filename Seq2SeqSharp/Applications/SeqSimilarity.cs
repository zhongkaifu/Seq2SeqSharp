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

        public SeqSimilarity(SeqSimilarityOptions options, Vocab srcVocab = null, Vocab clsVocab = null)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;

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
                    encoderType, srcVocab, clsVocab, options.EnableSegmentEmbeddings);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();

            Logger.WriteLine($"Max source sentence length in training corpus = '{options.MaxTrainSentLength}'");
            Logger.WriteLine($"BeamSearch Size = '{options.BeamSearchSize}'");
        }


        public void Train(int maxTrainingEpoch, SeqClassificationMultiTasksCorpus trainCorpus, SeqClassificationMultiTasksCorpus validCorpus, ILearningRate learningRate, Dictionary<int, List<IMetric>> taskId2metrics, IOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, taskId2metrics, m_modelMetaData, RunForwardOnSingleDevice);
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
            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, m_options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: true, learningRateFactor: m_options.EncoderStartLearningRateFactor), DeviceIds);
                contextDim = modelMetaData.HiddenDim;
            }

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

            (IEncoder encoder, IWeightTensor srcEmbedding, IFeedForwardLayer encoderFFLayer, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            IWeightTensor encOutput1 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, m_shuffleType, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 0);
            IWeightTensor encOutput2 = Encoder.BuildTensorForSourceTokenGroupAt(computeGraph, sntPairBatch, ShuffleEnums.Random, encoder, m_modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding, 1);
            IWeightTensor encOutput = computeGraph.EltMul(encOutput1, encOutput2);
            IWeightTensor ffLayer = encoderFFLayer.Process(encOutput, batchSize, computeGraph);

            float cost = 0.0f;
            NetworkResult nr = new NetworkResult
            {
                Output = new List<List<List<string>>>()
            };

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

            nrs.Add(nr);

            return nrs;
        }
    }
}
