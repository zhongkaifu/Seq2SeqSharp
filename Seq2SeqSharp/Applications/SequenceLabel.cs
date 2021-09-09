using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
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
        private readonly int m_maxSntSize;

        public SequenceLabel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType,
            float dropoutRatio, Vocab srcVocab, Vocab clsVocab, int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath, int maxSntSize = 128) :
            base(deviceIds, processorType, modelFilePath)
        {
            m_modelMetaData = new SeqLabelModel(hiddenDim, embeddingDim, encoderLayerDepth, multiHeadNum, encoderType, srcVocab, clsVocab);
            m_dropoutRatio = dropoutRatio;
            m_maxSntSize = maxSntSize;

            //Initializng weights in encoders and decoders
            CreateTrainableParameters();

            m_modelMetaData.ShowModelInfo();
        }

        public SequenceLabel(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f, int maxSntSize = 128)
            : base(deviceIds, processorType, modelFilePath)
        {
            m_dropoutRatio = dropoutRatio;
            LoadModel(CreateTrainableParameters);
            m_maxSntSize = maxSntSize;

            m_modelMetaData.ShowModelInfo();
        }


        private bool CreateTrainableParameters()
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", m_modelMetaData.HiddenDim, m_modelMetaData.EncoderEmbeddingDim, m_modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", m_modelMetaData.HiddenDim * 2, m_modelMetaData.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", m_modelMetaData.MultiHeadNum, m_modelMetaData.HiddenDim, m_modelMetaData.EncoderEmbeddingDim, m_modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", m_modelMetaData.HiddenDim, m_modelMetaData.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { m_modelMetaData.SrcVocab.Count, m_modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SrcEmbeddings", isTrainable: true), DeviceIds);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(Math.Max(m_maxSntSize, m_maxSntSize) + 2, m_modelMetaData.EncoderEmbeddingDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), DeviceIds, true);
            }
            else
            {
                m_posEmbedding = null;
            }

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, IWeightTensor, FeedForwardLayer) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph g, ISntPairBatch sntPairBatch, int deviceIdIdx, bool isTraining)
        {
            List<NetworkResult> nrs = new List<NetworkResult>();

            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var tgtSnts = sntPairBatch.GetTgtTokens(0);

            (IEncoder encoder, IWeightTensor srcEmbedding, IWeightTensor posEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(g.GetWeightFactory(), srcSnts.Count);

            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
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
                                int ix_targets_k_j = m_modelMetaData.ClsVocab.GetWordIndex(tgtSnts[k][j]);
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
                    using var targetIdxTensor = g.Argmax(probs, 1);
                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    List<string> targetWords = m_modelMetaData.ClsVocab.ConvertIdsToString(targetIdx.ToList());

                    for (int k = 0; k < batchSize; k++)
                    {
                        tgtSnts[k] = targetWords.GetRange(k * seqLen, seqLen);
                    }
                }

            }

            NetworkResult nr = new NetworkResult
            {
                Cost = cost,
                Output = new List<List<List<string>>>()
            };
            nr.Output.Add(tgtSnts);

            nrs.Add(nr);

            return nrs;
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
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> srcSnts, IEncoder encoder, IWeightTensor Embedding, IWeightTensor srcSelfMask, IWeightTensor posEmbedding, float[] originalSrcLengths)
        {
            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            List<IWeightTensor> inputs = new List<IWeightTensor>();

            // Generate batch-first based input embeddings
            for (int j = 0; j < batchSize; j++)
            {
                int originalLength = (int)originalSrcLengths[j];
                for (int i = 0; i < seqLen; i++)
                {
                    int ix_source = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts[j][i], logUnk: true);

                    var emb = g.Peek(Embedding, 0, ix_source, runGradients: i < originalLength);

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
