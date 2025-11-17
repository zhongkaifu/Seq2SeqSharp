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
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using TensorSharp;

namespace Seq2SeqSharp.Applications
{
    public class VisionCaption : BaseSeq2SeqFramework<Seq2SeqModel>
    {
        private readonly Seq2SeqOptions m_options;
        private readonly VisionImageProcessor m_imageProcessor;
        private readonly PaddingEnums m_paddingType = PaddingEnums.AllowPadding;
        private readonly int m_visualTokenCount;

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder;
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder;
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer;
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_patchEmbedding;
        private MultiProcessorNetworkWrapper<INormalization> m_patchNorm;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_clsToken;
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_pointerGenerator;

        public VisionCaption(Seq2SeqOptions options, Vocab tgtVocab = null)
            : base(deviceIds: options.DeviceIds, processorType: options.ProcessorType, modelFilePath: options.ModelFilePath, memoryUsageRatio: options.MemoryUsageRatio,
                  compilerOptions: options.CompilerOptions, runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq,
                  startToRunValidAfterUpdates: options.StartValidAfterUpdates, maxDegressOfParallelism: options.TaskParallelism, mklInstructions: options.MKLInstructions,
                  weightsUpdateCount: options.WeightsUpdateCount, cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32,
                  saveModelEveryUpdats: options.SaveModelEveryUpdates, saveGPUMemoryLevel: options.SaveGPUMemoryLevel, initLossScaling: options.InitLossScaling,
                  autoCheckTensorCorruption: options.CheckTensorCorrupted, attentionType: options.AttentionType)
        {
            m_options = options;
            m_imageProcessor = new VisionImageProcessor(options);
            m_paddingType = options.PaddingType;
            m_visualTokenCount = m_imageProcessor.PatchCount + 1; // prepend CLS token

            m_options.ValidateOptions();

            if (File.Exists(options.ModelFilePath))
            {
                if (tgtVocab != null)
                {
                    throw new ArgumentException($"Model '{options.ModelFilePath}' already exists and contains vocabularies. Please do not pass new vocabularies while loading the model.");
                }

                m_modelMetaData = LoadModel();
            }
            else
            {
                m_modelMetaData = new Seq2SeqModel(options, null, tgtVocab);
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        public void UpdateVocabs(Vocab srcVocab, Vocab tgtVocab)
        {
            if (srcVocab != null)
            {
                Logger.WriteLine(Logger.Level.warn, "VisionCaption ignores the provided source vocabulary because images are tokenized on the fly.");
            }

            if (tgtVocab != null)
            {
                m_modelMetaData.TgtVocab = tgtVocab;
            }

            SaveModel(createBackupPrevious: true);
        }

        protected override Seq2SeqModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, Seq2SeqModel.Create, suffix);

        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine(Logger.Level.debug, "Creating encoder/decoder stacks for vision captioning...");

            var raDeviceIds = new RoundArray<int>(DeviceIds);
            DType elementType = m_options.AMP ? DType.Float16 : DType.Float32;

            m_encoder = Encoder.CreateEncoders(model, m_options, raDeviceIds, elementType: elementType);
            m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds, isTrainable: m_options.IsDecoderTrainable && (m_options.Task == ModeEnums.Train), elementType: elementType);
            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true, learningRateFactor: m_options.DecoderStartLearningRateFactor, elementType: elementType), DeviceIds);

            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, m_visualTokenCount, model, elementType: elementType, createAPE: (model.PEType == PositionEmbeddingEnums.APE));

            m_patchEmbedding = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("VisionPatchEmbedding", m_imageProcessor.FeatureSize, model.HiddenDim, m_options.DropoutRatio, raDeviceIds.GetNextItem(),
                isTrainable: m_options.IsSrcEmbeddingTrainable, learningRateFactor: m_options.EncoderStartLearningRateFactor, elementType: elementType), DeviceIds);
            m_patchNorm = new MultiProcessorNetworkWrapper<INormalization>(new LayerNormalization("VisionPatchNorm", model.HiddenDim, raDeviceIds.GetNextItem(), isTrainable: true,
                learningRateFactor: m_options.EncoderStartLearningRateFactor, elementType: elementType), DeviceIds);
            m_clsToken = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { 1, model.HiddenDim }, raDeviceIds.GetNextItem(),
                initType: RandomInitType.Uniform, name: "VisionCLSToken", isTrainable: true, learningRateFactor: m_options.EncoderStartLearningRateFactor, dtype: elementType), DeviceIds);

            m_tgtEmbedding = CreateTgtEmbeddings(model, raDeviceIds, m_options.IsTgtEmbeddingTrainable, m_options.DecoderStartLearningRateFactor, elementType);

            if (model.PointerGenerator)
            {
                throw new ArgumentException("Pointer generator is not supported for the vision caption pipeline.");
            }

            m_pointerGenerator = null;

            return true;
        }

        private (IEncoder, IDecoder, IFeedForwardLayer, IFeedForwardLayer, IWeightTensor, IWeightTensor, IFeedForwardLayer, IWeightTensor, INormalization, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_patchEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx),
                    m_pointerGenerator?.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx),
                    m_patchNorm?.GetNetworkOnDevice(deviceIdIdx),
                    m_clsToken?.GetNetworkOnDevice(deviceIdIdx));
        }

        private IWeightTensor BuildVisionEmbeddings(IComputeGraph computeGraph, List<string> srcImagePaths, IFeedForwardLayer patchEmbedding, INormalization patchNorm, IWeightTensor posEmbeddings, IWeightTensor clsToken)
        {
            var features = m_imageProcessor.BuildPatchFeatures(srcImagePaths);
            var factory = computeGraph.GetWeightFactory();
            var dtype = m_options.AMP ? DType.Float16 : DType.Float32;
            var featureTensor = factory.CreateWeightTensor(srcImagePaths.Count * m_imageProcessor.PatchCount, m_imageProcessor.FeatureSize, computeGraph.DeviceId, dtype,
                name: $"VisionPatchTensor_{Guid.NewGuid()}", isTrainable: false, graphToBind: computeGraph, needGradient: false);
            featureTensor.SetWeightArray(features);

            var embeddings = patchEmbedding.Process(featureTensor, srcImagePaths.Count, computeGraph);
            embeddings = patchNorm?.Norm(embeddings, computeGraph) ?? embeddings;
            embeddings = computeGraph.Mul(embeddings, (float)Math.Sqrt(embeddings.Columns), inPlace: true);

            int batchSize = srcImagePaths.Count;
            int hiddenDim = embeddings.Columns;
            embeddings = computeGraph.View(embeddings, dims: new long[] { batchSize, m_imageProcessor.PatchCount, hiddenDim });

            if (clsToken != null)
            {
                using var clsView = computeGraph.View(clsToken, dims: new long[] { 1, 1, hiddenDim });
                using var clsBatch = computeGraph.Expand(clsView, dims: new long[] { batchSize, 1, hiddenDim });
                embeddings = computeGraph.Concate(new List<IWeightTensor> { clsBatch, embeddings }, 1);
            }

            embeddings = computeGraph.View(embeddings, dims: new long[] { batchSize * m_visualTokenCount, hiddenDim });

            if (posEmbeddings != null)
            {
                embeddings = PositionEmbedding.AddPositionEmbedding(computeGraph, posEmbeddings, batchSize, embeddings, m_options.DropoutRatio);
            }

            return embeddings;
        }

        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, IPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            if (sntPairBatch is not VisionTextCorpusBatch visionBatch)
            {
                throw new ArgumentException("VisionCaption expects VisionTextCorpusBatch instances.");
            }

            visionBatch.SrcTokenCount = visionBatch.BatchSize * m_visualTokenCount;

            (var encoder, var decoder, var decoderFFLayer, var patchEmbedding, var tgtEmbedding, var segmentEmbedding, var pointerGenerator, var posEmbeddings, var patchNorm, var clsToken) = GetNetworksOnDeviceAt(computeGraph.DeviceId);

            if (pointerGenerator != null)
            {
                throw new InvalidOperationException("Pointer generator is disabled for vision captioning, but a pointer generator network was found.");
            }

            var encInput = BuildVisionEmbeddings(computeGraph, visionBatch.SrcBatchPaths, patchEmbedding, patchNorm, posEmbeddings, clsToken);
            var srcLengths = BuildSrcLengths(visionBatch.BatchSize);

            encoder.Reset(computeGraph.GetWeightFactory(), visionBatch.BatchSize);
            var encOutput = encoder.Encode(encInput, visionBatch.BatchSize, computeGraph, null);

            var tgtSnts = sntPairBatch.GetTgtTokens();
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);

            NetworkResult nr = new NetworkResult
            {
                Status = NetworkResultStatus.SUCCEED
            };

            decoder.Reset(computeGraph.GetWeightFactory(), visionBatch.BatchSize);

            if (decoder is AttentionDecoder attentionDecoder)
            {
                nr.Cost = Decoder.DecodeAttentionLSTM(tgtTokensList, computeGraph, encOutput, attentionDecoder, decoderFFLayer, tgtEmbedding, m_modelMetaData.TgtVocab, visionBatch.BatchSize, isTraining);
                nr.Output = new List<List<List<string>>>
                {
                    m_modelMetaData.TgtVocab.ConvertIdsToString(tgtTokensList)
                };
            }
            else
            {
                if (isTraining)
                {
                    (var c, _) = Decoder.DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, srcLengths, m_modelMetaData.TgtVocab, m_paddingType,
                        m_options.DropoutRatio, decodingOptions, isTraining, pointerGenerator: null, srcSeqs: null, lossType: m_options.LossType, labelSmooth: m_options.LabelSmooth,
                        segmentEmbeddings: segmentEmbedding, amp: m_options.AMP, posEmbeddings: posEmbeddings, lossScaling: LossScaling, paddingAlignmentFactor: m_options.PaddingAlignmentFactor);
                    nr.Cost = c;
                    nr.Output = null;
                }
                else if (m_options.Task == ModeEnums.Alignment)
                {
                    throw new ArgumentException("Alignment mode is not supported for the vision caption pipeline.");
                }
                else
                {
                    string cacheKey = GenerateCacheKey(tgtSnts);
                    Dictionary<string, IWeightTensor> cachedTensors = null;
                    if (m_options.UseKVCache)
                    {
                        cachedTensors = System.Runtime.Caching.MemoryCache.Default[cacheKey] as Dictionary<string, IWeightTensor>;
                        if (cachedTensors == null && decodingOptions.BeamSearchSize == 1)
                        {
                            cachedTensors = new Dictionary<string, IWeightTensor>();
                        }
                        System.Runtime.Caching.MemoryCache.Default.Remove(cacheKey);
                    }

                    List<List<BeamSearchStatus>> beam2batchStatus = Decoder.InitBeamSearchStatusListList(visionBatch.BatchSize, tgtTokensList);
                    for (int i = tgtTokensList[0].Count; i < decodingOptions.MaxTgtSentLength; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = null;
                        try
                        {
                            foreach (var batchStatus in beam2batchStatus)
                            {
                                var batch2tgtTokens = Decoder.ExtractBatchTokens(batchStatus);
                                using var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}");
                                (var cost2, var bssSeqList) = Decoder.DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding,
                                    srcLengths, m_modelMetaData.TgtVocab, m_paddingType, 0.0f, decodingOptions, isTraining, outputSentScore: decodingOptions.BeamSearchSize > 1,
                                    previousBeamSearchResults: batchStatus, pointerGenerator: null, srcSeqs: null, contextTensors: cachedTensors, blockedTokens: decodingOptions.BlockedTokens,
                                    segmentEmbeddings: segmentEmbedding, amp: m_options.AMP, posEmbeddings: posEmbeddings, lossScaling: LossScaling, paddingAlignmentFactor: m_options.PaddingAlignmentFactor);

                                _ = cost2;
                                bssSeqList = Decoder.SwapBeamAndBatch(bssSeqList);
                                batch2beam2seq = Decoder.CombineBeamSearchResults(batch2beam2seq, bssSeqList);
                            }
                        }
                        catch (OutOfMemoryException)
                        {
                            GC.Collect();
                            Logger.WriteLine(Logger.Level.warn, $"Out of memory while generating token index '{i}'. Terminating decoding for current batch.");
                            nr.Status = NetworkResultStatus.OOM;
                            break;
                        }

                        if (decodingOptions.BeamSearchSize > 1)
                        {
                            for (int k = 0; k < visionBatch.BatchSize; k++)
                            {
                                batch2beam2seq[k] = BeamSearch.GetTopNBSS(batch2beam2seq[k], decodingOptions.BeamSearchSize);
                            }
                        }

                        beam2batchStatus = Decoder.SwapBeamAndBatch(batch2beam2seq);
                        if (Decoder.AreAllSentsCompleted(beam2batchStatus))
                        {
                            break;
                        }
                    }

                    nr.Cost = 0.0f;
                    nr.Output = m_modelMetaData.TgtVocab.CovertToWords(beam2batchStatus);

                    if (cachedTensors != null)
                    {
                        cacheKey = GenerateCacheKey(nr.Output[0]);
                        Dictionary<string, IWeightTensor> newCachedTensors = new Dictionary<string, IWeightTensor>();
                        foreach (var pair in cachedTensors)
                        {
                            newCachedTensors.Add(pair.Key, pair.Value.CopyWeightsRef(pair.Value.Name, false, graphToBind: null));
                        }
                        System.Runtime.Caching.MemoryCache.Default.Set(cacheKey, newCachedTensors, DateTimeOffset.Now + TimeSpan.FromMinutes(10));
                    }
                }
            }

            nr.RemoveDuplicatedEOS();

            return new List<NetworkResult> { nr };
        }

        private float[] BuildSrcLengths(int batchSize)
        {
            var lengths = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                lengths[i] = m_visualTokenCount;
            }

            return lengths;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            if (!outputSrcVocab.IsNullOrEmpty())
            {
                Logger.WriteLine(Logger.Level.warn, "VisionCaption does not maintain a source vocabulary. The requested source vocab dump will be skipped.");
            }

            if (!outputTgtVocab.IsNullOrEmpty())
            {
                m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
            }
        }

        public void Test(string inputTestFile, string outputFile, int batchSize, DecodingOptions decodingOptions)
        {
            Test<VisionTextCorpusBatch>(inputTestFile, outputFile, batchSize, decodingOptions, srcSpmPath: null, tgtSpmPath: null);
        }

        public void VQModel()
        {
            m_modelMetaData.VQType = m_options.VQType;
            SaveModel(createBackupPrevious: true, suffix: ".vq");
        }

        private static string GenerateCacheKey(List<List<string>> sequences)
        {
            var parts = new List<string>(sequences.Count);
            foreach (var seq in sequences)
            {
                parts.Add(string.Join(" ", seq));
            }

            return string.Join("\t", parts);
        }
    }
}
