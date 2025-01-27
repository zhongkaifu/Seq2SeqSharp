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
using AdvUtils;
using System.Runtime.Caching;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using TensorSharp;
using ManagedCuda.BasicTypes;

namespace Seq2SeqSharp.Applications
{
    public class DPO : BaseSeq2SeqFramework<Seq2SeqModel>
    {
        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding = null; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder = null; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer = null; //The feed forward layers over devices after all layers in decoder
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding = null;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding = null;






        private MultiProcessorNetworkWrapper<IWeightTensor> ref_m_tgtEmbedding = null; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IDecoder> ref_m_decoder = null; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> ref_m_decoderFFLayer = null; //The feed forward layers over devices after all layers in decoder
        private MultiProcessorNetworkWrapper<IWeightTensor> ref_m_segmentEmbedding = null;
        private MultiProcessorNetworkWrapper<IWeightTensor> ref_m_posEmbedding = null;




        private readonly PaddingEnums m_paddingType = PaddingEnums.AllowPadding;
        readonly Seq2SeqOptions m_options = null;

        public event EventHandler KVCacheRemoveWatcher;

        public DPO(Seq2SeqOptions options, Vocab tgtVocab = null)
            : base(deviceIds: options.DeviceIds, processorType: options.ProcessorType, modelFilePath: options.ModelFilePath, memoryUsageRatio: options.MemoryUsageRatio,
                  compilerOptions: options.CompilerOptions, runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq,
                  startToRunValidAfterUpdates: options.StartValidAfterUpdates, maxDegressOfParallelism: options.TaskParallelism, mklInstructions: options.MKLInstructions, weightsUpdateCount: options.WeightsUpdateCount,
                  enableTensorCore: options.EnableTensorCore, cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32, randomSeed: options.RandomSeed,
                  saveModelEveryUpdats: options.SaveModelEveryUpdates, saveGPUMemoryLevel: options.SaveGPUMemoryLevel, initLossScaling: options.InitLossScaling, autoCheckTensorCorruption: options.CheckTensorCorrupted,
                  attentionType: options.AttentionType)
        {
            m_paddingType = options.PaddingType;
            m_options = options;

            // Check if options are valided.
            m_options.ValidateOptions();
            if (File.Exists(m_options.ModelFilePath))
            {
                if (tgtVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                // Model file exists, so we load it from file.
                m_modelMetaData = LoadModel();
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                m_modelMetaData = new Seq2SeqModel(options, null, tgtVocab);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.EncoderType = EncoderTypeEnums.None;
            m_modelMetaData.DecoderType = DecoderTypeEnums.GPTDecoder;
            m_modelMetaData.ShowModelInfo();
        }

        public void UpdateVocabs(Vocab tgtVocab)
        {
            if (tgtVocab != null)
            {
                m_modelMetaData.TgtVocab = tgtVocab;
            }

            SaveModel(createBackupPrevious: true, suffix: ".updatevocab");
        }

        public void VQModel()
        {
            m_modelMetaData.VQType = m_options.VQType;
            SaveModel(createBackupPrevious: true, suffix: $".{m_modelMetaData.VQType.ToString()}");

        }

        protected override Seq2SeqModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, Seq2SeqModel.Create, suffix);

        private bool CreateTrainableParameters(IModel model)
        {
            CreateDPOModel(model);
            CreateRefModel(model);

            return true;
        }

        private bool CreateDPOModel(IModel model)
        {
            if (m_decoder != null)
            {
                m_decoder.Dispose();
            }
            if (m_decoderFFLayer != null)
            {
                m_decoderFFLayer.Dispose();
            }

            if (m_segmentEmbedding != null)
            {
                m_segmentEmbedding.Dispose();
            }

            if (m_tgtEmbedding != null)
            {
                m_tgtEmbedding.Dispose();
            }

            Logger.WriteLine(Logger.Level.debug, $"Creating decoders...");

            var raDeviceIds = new RoundArray<int>(DeviceIds);

            DType elementType = m_options.AMP ? DType.Float16 : DType.Float32;

            m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds, isTrainable: m_options.IsDecoderTrainable && (m_options.Task == ModeEnums.DPO), elementType: elementType);
            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: (m_options.Task == ModeEnums.DPO), learningRateFactor: m_options.DecoderStartLearningRateFactor, elementType), DeviceIds);

            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, Math.Max(m_options.MaxTgtSentLength, m_options.MaxValidTgtSentLength), model, elementType,
                isTrainable: (m_options.Task == ModeEnums.DPO), createAPE: (model.PEType == PositionEmbeddingEnums.APE));
            m_tgtEmbedding = CreateTgtEmbeddings(model, raDeviceIds, m_options.IsTgtEmbeddingTrainable && (m_options.Task == ModeEnums.DPO), m_options.DecoderStartLearningRateFactor, elementType);

            return (true);
        }


        private bool CreateRefModel(IModel model)
        {
            if (ref_m_decoder != null)
            {
                ref_m_decoder.Dispose();
            }
            if (ref_m_decoderFFLayer != null)
            {
                ref_m_decoderFFLayer.Dispose();
            }

            if (ref_m_segmentEmbedding != null)
            {
                ref_m_segmentEmbedding.Dispose();
            }

            if (ref_m_tgtEmbedding != null)
            {
                ref_m_tgtEmbedding.Dispose();
            }

            Logger.WriteLine(Logger.Level.debug, $"Creating decoders...");

            var raDeviceIds = new RoundArray<int>(DeviceIds);

            DType elementType = m_options.AMP ? DType.Float16 : DType.Float32;

            ref_m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds, isTrainable: false, elementType: elementType, isSavable: false);
            ref_m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: false, learningRateFactor: m_options.DecoderStartLearningRateFactor, elementType), DeviceIds, savableWeights: false);

            (ref_m_posEmbedding, ref_m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, Math.Max(m_options.MaxTgtSentLength, m_options.MaxValidTgtSentLength), model, elementType,
                isTrainable: false, createAPE: (model.PEType == PositionEmbeddingEnums.APE));
            ref_m_tgtEmbedding = CreateTgtEmbeddings(model, raDeviceIds, false, m_options.DecoderStartLearningRateFactor, elementType, isSavable: false);

            return (true);
        }
        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        private (IDecoder, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx));
        }


        private (IDecoder, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor) GetRefNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (ref_m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    ref_m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    ref_m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    ref_m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx),
                    ref_m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, IPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            if (isTraining == false)
            {
                throw new ArgumentException("The DPO is only for training mode.");
            }

            (var decoder, var decoderFFLayer, var tgtEmbedding, var segmentEmbedding, var posEmbeddings) = GetNetworksOnDeviceAt(computeGraph.DeviceId);
            (var ref_decoder, var ref_decoderFFLayer, var ref_tgtEmbedding, var ref_segmentEmbedding, var ref_posEmbeddings) = GetRefNetworksOnDeviceAt(computeGraph.DeviceId);

            List<NetworkResult> nrs = new List<NetworkResult>();
            int messageTokenId = m_modelMetaData.TgtVocab.GetWordIndex(m_options.DPOMaskedToken, logUnk: true);

            // Generate output decoder sentences
            var chosenSnts = sntPairBatch.GetSrcTokens();
            int batchSize = chosenSnts.Count;
            var chosenTokensList = m_modelMetaData.TgtVocab.GetWordIndex(chosenSnts);
            var chosenMask = computeGraph.BuildMaskAfter(chosenTokensList, messageTokenId, tgtEmbedding.ElementType);


            var rejectedSnts = sntPairBatch.GetTgtTokens();
            //int batchSize = preferredSnts.Count;
            var rejectedTokensList = m_modelMetaData.TgtVocab.GetWordIndex(rejectedSnts);
            var rejectedMask = computeGraph.BuildMaskAfter(rejectedTokensList, messageTokenId, tgtEmbedding.ElementType);

            NetworkResult nr = new NetworkResult();
            nr.Status = NetworkResultStatus.SUCCEED;

            decoder.Reset(computeGraph.GetWeightFactory(), chosenSnts.Count);
            //decoder.Reset(computeGraph.GetWeightFactory(), nonPreferredSnts.Count);

            (var loss, var cr, var rr) = Decoder.DPODecoderTrainer(chosenTokensList, rejectedTokensList, computeGraph, decoder as GPTDecoder, ref_decoder as GPTDecoder, 
                decoderFFLayer, ref_decoderFFLayer,
                tgtEmbedding, ref_tgtEmbedding,
                m_modelMetaData.TgtVocab, m_paddingType, m_options.DropoutRatio, 
                segmentEmbedding, ref_segmentEmbedding,
                m_options.AMP, 
                posEmbeddings, ref_posEmbeddings,
                LossScaling, m_options.PaddingAlignmentFactor, lossSmooth: m_options.LossSmooth, beta: m_options.DPOBeta, chosenMasks: chosenMask, rejectedMasks: rejectedMask);
            nr.Cost = loss;
            nr.ChosenRewards = cr;
            nr.RejectedRewards = rr;
            nr.Output = null;

            nrs.Add(nr);
            return nrs;
        }
    }
}
