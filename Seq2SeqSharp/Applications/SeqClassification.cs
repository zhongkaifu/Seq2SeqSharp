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
using TensorSharp;

namespace Seq2SeqSharp.Applications
{
    public class SeqClassification : BaseSeq2SeqFramework<SeqClassificationModel>
    {
        public Vocab SrcVocab => m_modelMetaData.SrcVocab;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder

        public Vocab TgtVocab => m_modelMetaData.TgtVocab;

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_positionalEmbeddings = null;

        private readonly PaddingEnums m_paddingType = PaddingEnums.AllowPadding;
        private readonly SeqClassificationOptions m_options;

        public SeqClassification(SeqClassificationOptions options, Vocab srcVocab = null, Vocab tgtVocab = null)
           : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions,
                 runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq, maxDegressOfParallelism: options.TaskParallelism, 
                 cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32, saveModelEveryUpdats: options.SaveModelEveryUpdates, 
                 initLossScaling: options.InitLossScaling, autoCheckTensorCorruption: options.CheckTensorCorrupted)
        {
            m_paddingType = options.PaddingType;
            m_options = options;

            // Check if options are valided.
            m_options.ValidateOptions();

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || tgtVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                m_modelMetaData = LoadModel();
            }
            else
            {
                m_modelMetaData = new SeqClassificationModel(options, srcVocab, tgtVocab);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        protected override SeqClassificationModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, SeqClassificationModel.Create, suffix);
        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine(Logger.Level.debug, $"Creating encoders...");

            var raDeviceIds = new RoundArray<int>(DeviceIds);
            m_encoder = Encoder.CreateEncoders(model, m_options, raDeviceIds);
            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer($"FeedForward_Encoder", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);


            (m_positionalEmbeddings, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, m_options.MaxSentLength, model, createAPE: (model.PEType == PositionEmbeddingEnums.APE));

            Logger.WriteLine(Logger.Level.debug, $"Creating embeddings. Shape = '({model.SrcVocab.Count} ,{model.EncoderEmbeddingDim})'");

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { model.SrcVocab.Count, model.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, fanOut: true, name: "SrcEmbeddings",
                isTrainable: m_options.IsEmbeddingTrainable), DeviceIds);

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IWeightTensor, IFeedForwardLayer, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_encoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_positionalEmbeddings?.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, IPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            List<NetworkResult> nrs = new List<NetworkResult>();

            (IEncoder encoder, IWeightTensor srcEmbedding, var encoderFFLayer, IWeightTensor segmentEmbedding, IWeightTensor posEmbeddings) = GetNetworksOnDeviceAt(computeGraph.DeviceId);
            var srcSnts = sntPairBatch.GetSrcTokens();
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);
            IWeightTensor encOutput = Encoder.Run(computeGraph, encoder, m_modelMetaData, m_paddingType, srcEmbedding, posEmbeddings, segmentEmbedding, srcTokensList, originalSrcLengths);

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

            var indice = computeGraph.CreateTensorWeights(new long[] { batchSize, 1 }, clsIdxs);
            IWeightTensor clsWeightTensor = computeGraph.IndexSelect(encOutput, indice);

            NetworkResult nr = new NetworkResult
            {
                Output = new List<List<List<string>>>()
            };

            IWeightTensor ffLayer = encoderFFLayer.Process(clsWeightTensor, batchSize, computeGraph);
            IWeightTensor probs = computeGraph.Softmax(ffLayer, inPlace: true);

            if (isTraining)
            {
                var tgtSnts = sntPairBatch.GetTgtTokens();
                var tgtTokensLists = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);
                var tgtTokensTensor = computeGraph.CreateTokensTensor(tgtTokensLists);
                nr.Cost = computeGraph.CrossEntropyLoss(probs, tgtTokensTensor, graident: LossScaling);
            }
            else
            {
                // Output "i"th target word
                using var targetIdxTensor = computeGraph.Argmax(probs, 1);
                float[] targetIdx = targetIdxTensor.ToWeightArray();
                List<string> targetWords = m_modelMetaData.TgtVocab.ConvertIdsToString(targetIdx.ToList());
                nr.Output.Add(new List<List<string>>());

                for (int k = 0; k < batchSize; k++)
                {
                    nr.Output[0].Add(new List<string>());
                    nr.Output[0][k].Add(targetWords[k]);
                }
            }

            nrs.Add(nr);

            return nrs;
        }
    }
}
