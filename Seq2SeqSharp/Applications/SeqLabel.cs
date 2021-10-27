using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp
{
    public class SeqLabel : BaseSeq2SeqFramework
    {
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_ffLayer; //The feed forward layers over over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        private readonly SeqLabelOptions m_options;

        public SeqLabel(SeqLabelOptions options, Vocab srcVocab = null, Vocab clsVocab = null)
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
                if (srcVocab != null || clsVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                // Model file exists, so we load it from file.
                LoadModel(CreateTrainableParameters);
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);
                m_modelMetaData = new SeqLabelModel(options.HiddenSize, options.EmbeddingDim, options.EncoderLayerDepth, options.MultiHeadNum, encoderType, srcVocab, clsVocab, options.MaxSegmentNum);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters();
            }

            m_modelMetaData.ShowModelInfo();
        }


        private bool CreateTrainableParameters()
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(m_modelMetaData, m_options, raDeviceIds);
            m_ffLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", contextDim, m_modelMetaData.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { m_modelMetaData.SrcVocab.Count, m_modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SrcEmbeddings", isTrainable: true), DeviceIds);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(m_options.MaxTestSentLength + 2, m_modelMetaData.EncoderEmbeddingDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), DeviceIds, true);
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
                m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_ffLayer.GetNetworkOnDevice(deviceIdIdx));
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
            BuildInTokens.PadSentences(tgtSnts);
            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            // Encoding input source sentences
            IWeightTensor encOutput = Encoder.Run(g, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, null, srcSnts, originalSrcLengths);
            IWeightTensor ffLayer = decoderFFLayer.Process(encOutput, batchSize, g);

            float cost = 0.0f;
            using (IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        for (int j = 0; j < seqLen; j++)
                        {

                            if (k >= tgtSnts.Count)
                            {
                                throw new IndexOutOfRangeException($"Sequence #'{k}' is out of range in target sequences (size '{tgtSnts.Count})'. Source sequences batch size is '{srcSnts.Count}'");
                            }

                            if (j >= tgtSnts[k].Count)
                            {
                                throw new IndexOutOfRangeException($"Token offset '{j}' is out of range in current target sequence (size = '{tgtSnts[k].Count}' text = '{String.Join(' ',tgtSnts[k])}'). Source sequence size is '{srcSnts[k].Count}' text is {String.Join(' ', srcSnts[k])}");
                            }


                            int ix_targets_k_j = m_modelMetaData.ClsVocab.GetWordIndex(tgtSnts[k][j]);
                            float score_k = probs.GetWeightAt(new long[] { k * seqLen + j, ix_targets_k_j });
                            cost += (float)-Math.Log(score_k);

                            probs.SetWeightAt(score_k - 1, new long[] { k * seqLen + j, ix_targets_k_j });
                        }
                    }

                    ffLayer.CopyWeightsToGradients(probs);
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
    }
}
