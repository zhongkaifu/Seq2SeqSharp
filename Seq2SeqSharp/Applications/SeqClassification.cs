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
    public class SeqClassification : BaseSeq2SeqFramework
    {
        public Vocab SrcVocab => m_modelMetaData.SrcVocab;
        public List<Vocab> ClsVocabs => m_modelMetaData.ClsVocabs;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IFeedForwardLayer>[] m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        readonly SeqClassificationOptions m_options = null;

        public SeqClassification(SeqClassificationOptions options, Vocab srcVocab = null, List<Vocab> clsVocabs = null)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours, updateFreq: options.UpdateFreq)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;

            // Model must exist if current task is not for training
            if (m_options.Task.Equals("Train", StringComparison.InvariantCultureIgnoreCase) == false && File.Exists(m_options.ModelFilePath) == false)
            {
                throw new FileNotFoundException($"Model '{m_options.ModelFilePath}' doesn't exist.");
            }

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || clsVocabs != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                LoadModel(CreateTrainableParameters);
            }
            else
            {
                EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);

                m_modelMetaData = new SeqClassificationModel(options.HiddenSize, options.EmbeddingDim, options.EncoderLayerDepth, options.MultiHeadNum,
                    encoderType, srcVocab, clsVocabs, options.EnableSegmentEmbeddings, options.ApplyContextEmbeddingsToEntireSequence);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters();
            }

            m_modelMetaData.ShowModelInfo();
        }

        private bool CreateTrainableParameters()
        {
            Logger.WriteLine($"Creating encoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(m_modelMetaData, m_options, raDeviceIds);

            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>[m_modelMetaData.ClsVocabs.Count];
            for (int i = 0; i < m_modelMetaData.ClsVocabs.Count; i++)
            {
                m_encoderFFLayer[i] = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer($"FeedForward_Encoder_{i}", contextDim, m_modelMetaData.ClsVocabs[i].Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                    Math.Max(m_options.MaxTrainSentLength, m_options.MaxTestSentLength) + 2,
                    contextDim, DeviceIds[0], "PosEmbedding", false), DeviceIds, true);

                if (m_modelMetaData.EnableSegmentEmbeddings)
                {
                    m_segmentEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { m_options.MaxSegmentSize, m_modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SegmentEmbedding", isTrainable: true), DeviceIds);
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

            Logger.WriteLine($"Creating embeddings. Shape = '({m_modelMetaData.SrcVocab.Count} ,{m_modelMetaData.EncoderEmbeddingDim})'");
            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { m_modelMetaData.SrcVocab.Count, m_modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsEmbeddingTrainable), DeviceIds);

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, List<IFeedForwardLayer>, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            List<IFeedForwardLayer> feedForwardLayers = new List<IFeedForwardLayer>();
            foreach (var item in m_encoderFFLayer)
            {
                feedForwardLayers.Add(item.GetNetworkOnDevice(deviceIdIdx));
            }

            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    feedForwardLayers,
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
            List<NetworkResult> nrs = new List<NetworkResult>();

            (IEncoder encoder, IWeightTensor srcEmbedding, List<IFeedForwardLayer> encoderFFLayer, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
          
            IWeightTensor encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, segmentEmbedding, srcSnts, originalSrcLengths);

            int srcSeqPaddedLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;
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
            for (int i = 0; i < m_encoderFFLayer.Length; i++)
            {
                float cost = 0.0f;
                NetworkResult nr = new NetworkResult
                {
                    Output = new List<List<List<string>>>()
                };

                IWeightTensor ffLayer = encoderFFLayer[i].Process(clsWeightTensor, batchSize, computeGraph);               
                using (IWeightTensor probs = computeGraph.Softmax(ffLayer, runGradients: false, inPlace: true))
                {
                    if (isTraining)
                    {
                        var tgtSnts = sntPairBatch.GetTgtTokens(i);
                        for (int k = 0; k < batchSize; k++)
                        {
                            int ix_targets_k_j = m_modelMetaData.ClsVocabs[i].GetWordIndex(tgtSnts[k][0]);
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
                        List<string> targetWords = m_modelMetaData.ClsVocabs[i].ConvertIdsToString(targetIdx.ToList());
                        nr.Output.Add(new List<List<string>>());

                        for (int k = 0; k < batchSize; k++)
                        {
                            nr.Output[0].Add(new List<string>());
                            nr.Output[0][k].Add(targetWords[k]);
                        }
                    }
                }

                nrs.Add(nr);
            }


            return nrs;

        }
    }
}
