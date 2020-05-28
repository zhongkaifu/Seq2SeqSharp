using AdvUtils;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Tools;
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
        private readonly float m_dropoutRatio;
        private readonly SeqLabelModelMetaData m_modelMetaData;
        private readonly object locker = new object();

        public SequenceLabel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType,
            float dropoutRatio, Vocab vocab, int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath) :
            base(deviceIds, processorType, modelFilePath)
        {
            m_modelMetaData = new SeqLabelModelMetaData(hiddenDim, embeddingDim, encoderLayerDepth, multiHeadNum, encoderType, vocab);
            m_dropoutRatio = dropoutRatio;

            //Initializng weights in encoders and decoders
            CreateTrainableParameters(m_modelMetaData);
        }

        public SequenceLabel(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, float dropoutRatio = 0.0f)
            : base(deviceIds, processorType, modelFilePath)
        {
            m_dropoutRatio = dropoutRatio;
            m_modelMetaData = LoadModel(CreateTrainableParameters) as SeqLabelModelMetaData;
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

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: NormType.Normal, name: "SrcEmbeddings", isTrainable: true), DeviceIds);
            //      m_crfDecoder = new CRFDecoder(modelMetaData.Vocab.TargetWordSize);

            return true;
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
        private (IEncoder, IWeightTensor, FeedForwardLayer) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx), m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private float RunForwardOnSingleDevice(IComputeGraph g, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {
            (IEncoder encoder, IWeightTensor srcEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(deviceIdIdx);
            int batchSize = srcSnts.Count;

            // Reset networks
            encoder.Reset(g.GetWeightFactory(), batchSize);

            // Encoding input source sentences
            ParallelCorpus.PadSentences(srcSnts);

            if (isTraining)
            {
                ParallelCorpus.PadSentences(tgtSnts);

                if (srcSnts[0].Count != tgtSnts[0].Count)
                {
                    throw new ArgumentException($"The length of source side and target side must be equal. source length = '{srcSnts[0].Count}', target length = '{tgtSnts[0].Count}'");
                }
            }
            int seqLen = srcSnts[0].Count;

            IWeightTensor encodedWeightMatrix = Encode(g, srcSnts, encoder, srcEmbedding);
            IWeightTensor ffLayer = decoderFFLayer.Process(encodedWeightMatrix, batchSize, g);

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
                            using (IWeightTensor probs_k_j = g.PeekRow(probs, k * seqLen + j, runGradients: false))
                            {
                                int ix_targets_k_j = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][j]);
                                float score_k = probs_k_j.GetWeightAt(ix_targets_k_j);
                                cost += (float)-Math.Log(score_k);

                                probs_k_j.SetWeightAt(score_k - 1, ix_targets_k_j);
                            }

                        }

                        ////CRF part
                        //using (var probs_k = g.PeekRow(probs, k * seqLen, seqLen, runGradients: false))
                        //{
                        //    var weights_k = probs_k.ToWeightArray();
                        //    var crfOutput_k = m_crfDecoder.ForwardBackward(seqLen, weights_k);

                        //    int[] trueTags = new int[seqLen];
                        //    for (int j = 0; j < seqLen; j++)
                        //    {
                        //        trueTags[j] = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][j]);
                        //    }
                        //    m_crfDecoder.UpdateBigramTransition(seqLen, crfOutput_k, trueTags);
                        //}

                    }

                    ffLayerBatch.CopyWeightsToGradients(probs);
                }
                else
                {
                    // CRF decoder
                    //for (int k = 0; k < batchSize; k++)
                    //{
                    //    //CRF part
                    //    using (var probs_k = g.PeekRow(probs, k * seqLen, seqLen, runGradients: false))
                    //    {
                    //        var weights_k = probs_k.ToWeightArray();

                    //        var crfOutput_k = m_crfDecoder.DecodeNBestCRF(weights_k, seqLen, 1);
                    //        var targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(crfOutput_k[0].ToList());
                    //        tgtSnts.Add(targetWords);
                    //    }
                    //}


                    // Output "i"th target word
                    int[] targetIdx = g.Argmax(probs, 1);
                    List<string> targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());

                    for (int k = 0; k < batchSize; k++)
                    {
                        tgtSnts.Add(targetWords.GetRange(k * seqLen, seqLen));
                    }
                }

            }

            return cost;
        }

        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        /// <param name="g"></param>
        /// <param name="inputSentences"></param>
        /// <param name="encoder"></param>
        /// <param name="reversEncoder"></param>
        /// <param name="Embedding"></param>
        /// <returns></returns>
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> inputSentences, IEncoder encoder, IWeightTensor Embedding)
        {
            int seqLen = inputSentences[0].Count;
            int batchSize = inputSentences.Count;

            List<IWeightTensor> forwardInput = new List<IWeightTensor>();
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < inputSentences.Count; j++)
                {
                    int ix_source = m_modelMetaData.Vocab.GetSourceWordIndex(inputSentences[j][i], logUnk: true);
                    forwardInput.Add(g.PeekRow(Embedding, ix_source));
                }
            }

            return encoder.Encode(g.ConcatRows(forwardInput), batchSize, g, null);
        }


        //internal override void SaveParameters(Stream stream)
        //{
        //    Logger.WriteLine($"Saving parameters for sequence labeling model.");
        //    base.SaveParameters(stream);
        //    m_crfDecoder.Save(stream);
        //}

        //internal override void LoadParameters(Stream stream)
        //{
        //    Logger.WriteLine($"Loading parameters for sequence labeling model.");
        //    base.LoadParameters(stream);
        //    m_crfDecoder.Load(stream);
        //}
    }
}
