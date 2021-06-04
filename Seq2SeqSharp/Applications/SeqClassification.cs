using AdvUtils;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class SeqClassification : BaseSeq2SeqFramework
    {
        private readonly SeqClassificationModelMetaData m_modelMetaData;
        public Vocab Vocab => m_modelMetaData.Vocab;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after all layers in encoder

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;

        SeqClassificationOptions m_options = null;

        public SeqClassification(SeqClassificationOptions options)
            : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours)
        {
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);
            m_options = options;
            m_modelMetaData = LoadModel(CreateTrainableParameters) as SeqClassificationModelMetaData;
        }

        public SeqClassification(SeqClassificationOptions options, Vocab vocab)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, options.ValidIntervalHours)
        {
            EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), options.EncoderType);
            m_shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), options.ShuffleType);

            m_options = options;
            m_modelMetaData = new SeqClassificationModelMetaData(options.HiddenSize, options.EmbeddingDim, options.EncoderLayerDepth, options.MultiHeadNum,
                encoderType, vocab, options.EnableSegmentEmbeddings);

            Logger.WriteLine($"Max source sentence length in training corpus = '{options.MaxTrainSentLength}'");
            Logger.WriteLine($"BeamSearch Size = '{options.BeamSearchSize}'");
            Logger.WriteLine($"Enable segment embeddings = '{options.EnableSegmentEmbeddings}'");

            //Initializng weights in encoders and decoders
            CreateTrainableParameters(m_modelMetaData);
        }


        public void Train(int maxTrainingEpoch, ParallelCorpus trainCorpus, ParallelCorpus validCorpus, ILearningRate learningRate, List<IMetric> metrics, IOptimizer optimizer)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpus, learningRate, optimizer, metrics, m_modelMetaData, validCorpus.SentTgtPrefix, RunForwardOnSingleDevice);
            }
        }

        public void Valid(ParallelCorpus validCorpus, List<IMetric> metrics, string hypPrefix)
        {
            RunValid(validCorpus, RunForwardOnSingleDevice, metrics, hypPrefix, true);
        }

        public (List<List<List<string>>>, List<List<List<Alignment>>>) Test(List<List<string>> inputTokens, int beamSearchSize, string hypPrefix)
        {
            return RunTest(inputTokens, beamSearchSize, hypPrefix, RunForwardOnSingleDevice);
        }

        private bool CreateTrainableParameters(IModelMetaData mmd)
        {
            Logger.WriteLine($"Creating encoders...");
            SeqClassificationModelMetaData modelMetaData = mmd as SeqClassificationModelMetaData;
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim = 0;
            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, m_options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: true), DeviceIds);
                contextDim = modelMetaData.HiddenDim;
            }

            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward", contextDim, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);


            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                m_posEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                    Math.Max(m_options.MaxTrainSentLength, m_options.MaxTestSentLength) + 2,
                    contextDim, DeviceIds[0], "PosEmbedding", false), DeviceIds, true);

                if (modelMetaData.EnableSegmentEmbeddings)
                {
                    m_segmentEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { 16, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SegmentEmbedding", isTrainable: true), DeviceIds);
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

            Logger.WriteLine($"Creating embeddings. Shape = '({modelMetaData.Vocab.SourceWordSize} ,{modelMetaData.EmbeddingDim})'");
            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsEmbeddingTrainable), DeviceIds);

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
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding == null ? null : m_posEmbedding.GetNetworkOnDevice(deviceIdIdx), m_segmentEmbedding == null ? null : m_segmentEmbedding.GetNetworkOnDevice(deviceIdIdx));
        }

        private IWeightTensor ExtractTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, List<int> seqOriginalLengths, IWeightTensor segmentEmbedding)
        {
            int batchSize = seqs.Count;
            int seqLen = seqs[0].Count;

            float[] idxs = new float[batchSize * seqLen];
            float[] segIdxs = new float[batchSize * seqLen];

            for (int i = 0; i < batchSize; i++)
            {
                int segIdx = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    idxs[i * seqLen + j] = seqs[i][j];
                    segIdxs[i * seqLen + j] = segIdx;

                    string token = m_modelMetaData.Vocab.GetSourceString(seqs[i][j]);
                    if (token == ParallelCorpus.SEP)
                    {
                        //A new segment
                        segIdx++;
                    }
                }
            }

            if (segmentEmbedding == null)
            {
                return g.IndexSelect(embeddingsTensor, idxs);
            }
            else
            {
                return g.Add(g.IndexSelect(embeddingsTensor, idxs), g.IndexSelect(segmentEmbedding, segIdxs));
            }
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
            var inputEmbs = ExtractTokensEmbeddings(seqs, g, embeddings, seqOriginalLengths, segmentEmbeddings);

            if (m_modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, m_options.DropoutRatio);
            }

            return encoder.Encode(inputEmbs, batchSize, g, selfMask);
        }


        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private NetworkResult RunForwardOnSingleDevice(IComputeGraph computeGraph, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx, bool isTraining)
        {
            NetworkResult nr = new NetworkResult();

            (IEncoder encoder, IWeightTensor srcEmbedding, IFeedForwardLayer decoderFFLayer, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            List<int> originalSrcLengths = ParallelCorpus.PadSentences(srcSnts);
            int srcSeqPaddedLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;
            IWeightTensor srcSelfMask = m_shuffleType == ShuffleEnums.NoPaddingInSrc ? null : computeGraph.BuildPadSelfMask(srcSeqPaddedLen, originalSrcLengths); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            var srcTokensList = m_modelMetaData.Vocab.GetSourceWordIndex(srcSnts);
            IWeightTensor encOutput = Encode(computeGraph, srcTokensList, encoder, srcEmbedding, srcSelfMask, posEmbedding, originalSrcLengths, segmentEmbedding);

            if (srcSelfMask != null)
            {
                srcSelfMask.Dispose();
            }

            //List<IWeightTensor> clsWeightTensors = new List<IWeightTensor>();
            float[] clsIdxs = new float[batchSize];

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < srcSnts[i].Count; j++)
                {
                    if (srcSnts[i][j] == "[CLS]")
                    {
                        clsIdxs[i] = i * srcSeqPaddedLen + j;
                        break;
                    }
                }
            }

            IWeightTensor clsWeightTensor = computeGraph.IndexSelect(encOutput, clsIdxs); 
            IWeightTensor ffLayer = decoderFFLayer.Process(clsWeightTensor, batchSize, computeGraph);

            float cost = 0.0f;
            using (IWeightTensor probs = computeGraph.Softmax(ffLayer, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    for (int k = 0; k < batchSize; k++)
                    {
                        int ix_targets_k_j = m_modelMetaData.Vocab.GetTargetWordIndex(tgtSnts[k][0]);
                        float score_k = probs.GetWeightAt(new long[] {k, ix_targets_k_j });
                        cost += (float)-Math.Log(score_k);

                        probs.SetWeightAt(score_k - 1, new long[] {k, ix_targets_k_j });

                    }

                    ffLayer.CopyWeightsToGradients(probs);
                }
                else
                {
                    // Output "i"th target word
                    using (var targetIdxTensor = computeGraph.Argmax(probs, 1))
                    {
                        float[] targetIdx = targetIdxTensor.ToWeightArray();
                        List<string> targetWords = m_modelMetaData.Vocab.ConvertTargetIdsToString(targetIdx.ToList());

                        for (int k = 0; k < batchSize; k++)
                        {
                            tgtSnts[k].Add(targetWords[k]);
                        }
                    }
                }

            }

            nr.Cost = cost;
            nr.Beam2Batch2Output = new List<List<List<string>>>();
            nr.Beam2Batch2Output.Add(tgtSnts);
            return nr;

        }
    }
}
