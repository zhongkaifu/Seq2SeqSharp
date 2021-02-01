using AdvUtils;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp
{
    public class SequenceLabel : BaseSeq2SeqFramework
    {
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after LSTM layers in decoder
                                                                                 //  private CRFDecoder m_crfDecoder;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;

        private readonly float m_dropoutRatio;
        private readonly SeqLabelModelMetaData m_modelMetaData;
        private readonly int m_maxSntSize;

        public SequenceLabel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType,
            float dropoutRatio, Vocab vocab, int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath, int maxSntSize = 128) :
            base(deviceIds, processorType, modelFilePath)
        {
            m_modelMetaData = new SeqLabelModelMetaData(hiddenDim, embeddingDim, encoderLayerDepth, multiHeadNum, encoderType, vocab);
            m_dropoutRatio = dropoutRatio;
            m_maxSntSize = maxSntSize;

            //Initializng weights in encoders and decoders
            CreateTrainableParameters(m_modelMetaData);
        }

        public SequenceLabel(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f, int maxSntSize = 128)
            : base(deviceIds, processorType, modelFilePath)
        {
            m_dropoutRatio = dropoutRatio;
            m_modelMetaData = LoadModel(CreateTrainableParameters) as SeqLabelModelMetaData;
            m_maxSntSize = maxSntSize;
        }


        private bool CreateTrainableParameters(IModelMetaData mmd)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            SeqLabelModelMetaData modelMetaData = mmd as SeqLabelModelMetaData;
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", modelMetaData.HiddenDim * 2, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", modelMetaData.HiddenDim, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SrcEmbeddings", isTrainable: true), DeviceIds);

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(Math.Max(m_maxSntSize, m_maxSntSize) + 2, modelMetaData.EmbeddingDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), DeviceIds, true);
            }
            else
            {
                m_posEmbedding = null;
            }

            return true;
        }

        public void Train(int maxTrainingEpoch, IEnumerable<SntPairBatch> trainCorpus, IEnumerable<SntPairBatch> validCorpus, ILearningRate learningRate, List<IMetric> metrics, AdamOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, metrics, m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        public void Valid(IEnumerable<SntPairBatch> validCorpus, List<IMetric> metrics)
        {
            RunValid(validCorpus, RunForwardOnSingleDevice, metrics, true);
        }

        public List<List<string>> Test(List<List<string>> inputTokens)
        {
            var rst = RunTest(inputTokens, RunForwardOnSingleDevice);

            return rst[0];
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, IWeightTensor, FeedForwardLayer) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                m_posEmbedding == null ? null : m_posEmbedding.GetNetworkOnDevice(deviceIdIdx), m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private NetworkResult RunForwardOnSingleDevice(IComputeGraph g, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {
            NetworkResult nr = new NetworkResult();

            (IEncoder encoder, IWeightTensor srcEmbedding, IWeightTensor posEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(g.GetWeightFactory(), srcSnts.Count);


            List<int> originalSrcLengths = ParallelCorpus.PadSentences(srcSnts);
            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            // Encoding input source sentences
            IWeightTensor encOutput = Encode(g, srcSnts, encoder, srcEmbedding, null, posEmbedding, originalSrcLengths);
            IWeightTensor ffLayer = decoderFFLayer.Process(encOutput, batchSize, g);
            IWeightTensor ffLayerBatch = g.TransposeBatch(ffLayer, batchSize);

            float cost = 0.0f;
            using (IWeightTensor probs = g.Softmax(ffLayerBatch, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        for (int j = 0; j < seqLen; j++)
                        {
                                int ix_targets_k_j = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][j]);
                                float score_k = probs.GetWeightAt(new long[] { k * seqLen + j, ix_targets_k_j });
                                cost += (float)-Math.Log(score_k);

                                probs.SetWeightAt(score_k - 1, new long[] { k * seqLen + j, ix_targets_k_j });
                        }
                    }

                    ffLayerBatch.CopyWeightsToGradients(probs);
                }
                else
                {
                    // Output "i"th target word
                    int[] targetIdx = g.Argmax(probs, 1);
                    List<string> targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());

                    for (int k = 0; k < batchSize; k++)
                    {
                        tgtSnts[k] = targetWords.GetRange(k * seqLen, seqLen);
                    }
                }

            }

            nr.Cost = cost;
            nr.Beam2Batch2Output = new List<List<List<string>>>();
            nr.Beam2Batch2Output.Add(tgtSnts);
            return nr;
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
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> srcSnts, IEncoder encoder, IWeightTensor Embedding, IWeightTensor srcSelfMask, IWeightTensor posEmbedding, List<int> originalSrcLengths)
        {
            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            List<IWeightTensor> inputs = new List<IWeightTensor>();

            // Generate batch-first based input embeddings
            for (int j = 0; j < batchSize; j++)
            {
                int originalLength = originalSrcLengths[j];
                for (int i = 0; i < seqLen; i++)
                {
                    int ix_source = m_modelMetaData.Vocab.GetSourceWordIndex(srcSnts[j][i], logUnk: true);

                    var emb = g.PeekRow(Embedding, ix_source, runGradients: i < originalLength ? true : false);

                    inputs.Add(emb);
                }
            }

            var inputEmbs = g.ConcatRows(inputs);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbedding, batchSize, inputEmbs, m_dropoutRatio);
            }


            return encoder.Encode(inputEmbs, batchSize, g, srcSelfMask);
        }
    }
}
