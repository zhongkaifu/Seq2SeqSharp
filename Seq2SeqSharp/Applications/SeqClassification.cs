// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Applications
{
    public class SeqClassification : BaseSeq2SeqFramework<SeqClassificationModel>
    {
        public Vocab SrcVocab => m_modelMetaData.SrcVocab;
        public List<Vocab> ClsVocabs => m_modelMetaData.ClsVocabs;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IFeedForwardLayer>[] m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        private readonly SeqClassificationOptions m_options;

        public SeqClassification(SeqClassificationOptions options, Vocab srcVocab = null, List<Vocab> clsVocabs = null)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions,
                 runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq, maxDegressOfParallelism: options.TaskParallelism)
        {
            m_shuffleType = options.ShuffleType;
            m_options = options;

            // Model must exist if current task is not for training
            if ((m_options.Task != ModeEnums.Train) && !File.Exists(m_options.ModelFilePath))
            {
                throw new FileNotFoundException($"Model '{m_options.ModelFilePath}' doesn't exist.");
            }

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || clsVocabs != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                m_modelMetaData = LoadModelImpl_WITH_CONVERT(CreateTrainableParameters);
                //m_modelMetaData = LoadModelImpl();
                //---LoadModel_As_BinaryFormatter( CreateTrainableParameters );
            }
            else
            {
                m_modelMetaData = new SeqClassificationModel(options.HiddenSize, options.EmbeddingDim, options.EncoderLayerDepth, options.MultiHeadNum,
                    options.EncoderType, srcVocab, clsVocabs, options.EnableSegmentEmbeddings, options.EnableTagEmbeddings, options.MaxSegmentNum);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        protected override SeqClassificationModel LoadModelImpl() => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, SeqClassificationModel.Create);
        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine($"Creating encoders...");
            var raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(model, m_options, raDeviceIds);

            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>[model.ClsVocabs.Count];
            for (int i = 0; i < model.ClsVocabs.Count; i++)
            {
                m_encoderFFLayer[i] = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer($"FeedForward_Encoder_{i}", contextDim, model.ClsVocabs[i].Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);
            }

            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, contextDim, m_options.MaxSentLength, model);

            Logger.WriteLine($"Creating embeddings. Shape = '({model.SrcVocab.Count} ,{model.EncoderEmbeddingDim})'");
            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { model.SrcVocab.Count, model.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: m_options.IsEmbeddingTrainable), DeviceIds);

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, List<IFeedForwardLayer>, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);

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
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, DecodingOptions decodingOptions)
        {
            var isTraining = (m_options.Task == ModeEnums.Train);
            List<NetworkResult> nrs = new List<NetworkResult>();

            (IEncoder encoder, IWeightTensor srcEmbedding, List<IFeedForwardLayer> encoderFFLayer, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(computeGraph.DeviceId);
            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);
            IWeightTensor encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, segmentEmbedding, srcTokensList, originalSrcLengths);

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
                NetworkResult nr = new NetworkResult
                {
                    Output = new List<List<List<string>>>()
                };

                IWeightTensor ffLayer = encoderFFLayer[i].Process(clsWeightTensor, batchSize, computeGraph);
                IWeightTensor probs = computeGraph.Softmax(ffLayer, inPlace: true);

                if (isTraining)
                {
                    var tgtSnts = sntPairBatch.GetTgtTokens(i);
                    var tgtTokensLists = m_modelMetaData.ClsVocabs[i].GetWordIndex(tgtSnts);
                    var tgtTokensTensor = computeGraph.CreateTokensTensor(tgtTokensLists);
                    nr.Cost = computeGraph.CrossEntropyLoss(probs, tgtTokensTensor);
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

                nrs.Add(nr);
            }
            return nrs;
        }
    }
}
