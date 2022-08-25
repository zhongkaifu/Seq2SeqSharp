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
using Microsoft.Extensions.Caching.Memory;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp
{
    public class Seq2Seq : BaseSeq2SeqFramework<Seq2SeqModel>
    {
        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after all layers in decoder

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;

        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_pointerGenerator;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        readonly Seq2SeqOptions m_options = null;

        private MemoryCache m_memoryCache;

        public Seq2Seq(Seq2SeqOptions options, Vocab srcVocab = null, Vocab tgtVocab = null)
            : base(deviceIds: options.DeviceIds, processorType: options.ProcessorType, modelFilePath: options.ModelFilePath, memoryUsageRatio: options.MemoryUsageRatio, 
                  compilerOptions: options.CompilerOptions, runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq, 
                  startToRunValidAfterUpdates: options.StartValidAfterUpdates, maxDegressOfParallelism: options.TaskParallelism, mklInstructions: options.MKLInstructions)
        {
            m_shuffleType = options.ShuffleType;
            m_options = options;

            m_memoryCache = new MemoryCache(new MemoryCacheOptions
            {
                SizeLimit = 1024
            });

            // Model must exist if current task is not for training
            if ((m_options.Task != ModeEnums.Train) && !File.Exists(m_options.ModelFilePath))
            {
                throw new FileNotFoundException($"Model '{m_options.ModelFilePath}' doesn't exist.");
            }

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || tgtVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                // Model file exists, so we load it from file.
                m_modelMetaData = LoadModelImpl_WITH_CONVERT(CreateTrainableParameters);
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                m_modelMetaData = new Seq2SeqModel(options.HiddenSize, options.SrcEmbeddingDim, options.TgtEmbeddingDim, options.EncoderLayerDepth, options.DecoderLayerDepth, options.MultiHeadNum,
                   options.EncoderType, options.DecoderType, srcVocab, tgtVocab, options.EnableCoverageModel, options.SharedEmbeddings, options.EnableSegmentEmbeddings, options.EnableTagEmbeddings, 
                   options.MaxSegmentNum, pointerGenerator: options.PointerGenerator, expertNum: options.ExpertNum);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        protected override Seq2SeqModel LoadModelImpl() => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, Seq2SeqModel.Create);

        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            var raDeviceIds = new RoundArray<int>(DeviceIds);

            int contextDim;
            (m_encoder, contextDim) = Encoder.CreateEncoders(model, m_options, raDeviceIds);
            m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds, contextDim);
            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true, learningRateFactor: m_options.DecoderStartLearningRateFactor), DeviceIds);
            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, contextDim, Math.Max(Math.Max(m_options.MaxSrcSentLength, m_options.MaxValidSrcSentLength), Math.Max(m_options.MaxTgtSentLength, m_options.MaxValidTgtSentLength)), model);
            (m_srcEmbedding, m_tgtEmbedding) = CreateSrcTgtEmbeddings(model, raDeviceIds, m_options.IsSrcEmbeddingTrainable, m_options.IsTgtEmbeddingTrainable, m_options.EncoderStartLearningRateFactor, m_options.DecoderStartLearningRateFactor);


            if (model.PointerGenerator)
            {
                if (model.SharedEmbeddings == false)
                {
                    throw new ArgumentException($"Shared embeddings is required to true for pointer generator.");
                }

                Logger.WriteLine($"Create pointer generator weights...");
                m_pointerGenerator = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("PointerGenerator_0", model.HiddenDim, 1, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true, learningRateFactor: m_options.DecoderStartLearningRateFactor), DeviceIds);
            }
            else
            {
                m_pointerGenerator = null;
            }

            return (true);
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        private (IEncoder, IDecoder, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor, IWeightTensor, IFeedForwardLayer) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_pointerGenerator?.GetNetworkOnDevice(deviceIdIdx));
        }

        private string GenerateCacheKey(List<List<string>> strs)
        {
            List<string> r = new List<string>();

            foreach (var str in strs)
            {
                r.Add(string.Join(" ", str));
            }

            return string.Join("\t", r);
        }


        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            (var encoder, var decoder, var decoderFFLayer, var srcEmbedding, var tgtEmbedding, var posEmbedding, var segmentEmbedding, var pointerGenerator) = GetNetworksOnDeviceAt(computeGraph.DeviceId);

            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);

            if (isTraining && srcSnts[0].Count > m_options.MaxSrcSentLength + 2)
            {
                throw new InvalidDataException($"The source sentence is too long. Its length = '{srcSnts[0].Count}', but MaxSrcSentLength is '{m_options.MaxSrcSentLength}'. The sentence is '{string.Join(" ", srcSnts[0])}'");
            }

            IWeightTensor encOutput;
            if (!isTraining && (m_options.ProcessorType == ProcessorTypeEnums.CPU))
            {
                // Try to get src tensor from cache
                string cacheKey = GenerateCacheKey(srcSnts);
                if (!m_memoryCache.TryGetValue(cacheKey, out encOutput))
                {
                    encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, segmentEmbedding, srcTokensList, originalSrcLengths);

                    var cacheEntryOptions = new MemoryCacheEntryOptions().SetSize(1);
                    m_memoryCache.Set(cacheKey, encOutput.CopyWeightsRef($"cache_{encOutput.Name}", false, graphToBind: null), cacheEntryOptions);
                }


            }
            else
            {
                // Compute src tensor
                encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, posEmbedding, segmentEmbedding, srcTokensList, originalSrcLengths);
            }

            List<NetworkResult> nrs = new List<NetworkResult>();

            // Generate output decoder sentences
            int batchSize = srcSnts.Count;
            var tgtSnts = sntPairBatch.GetTgtTokens(0);
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);
            NetworkResult nr = new NetworkResult();

            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            if (decoder is AttentionDecoder)
            {
                nr.Cost = Decoder.DecodeAttentionLSTM(tgtTokensList, computeGraph, encOutput, decoder as AttentionDecoder, decoderFFLayer, tgtEmbedding, m_modelMetaData.TgtVocab, srcSnts.Count, isTraining);
                nr.Output = new List<List<List<string>>>
                {
                    m_modelMetaData.TgtVocab.ConvertIdsToString(tgtTokensList)
                };
            }
            else
            {
                if (isTraining)
                {
                    (var c, _) = Decoder.DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding, originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType,
                        m_options.DropoutRatio, decodingOptions, isTraining, pointerGenerator: pointerGenerator, srcSeqs: srcTokensList);
                    nr.Cost = c;
                    nr.Output = null;
                }
                else if (m_options.Task == ModeEnums.Alignment)
                {
                    if (decodingOptions.OutputAligmentsToSrc)
                    {
                        if (pointerGenerator == null)
                        {
                            throw new ArgumentException($"Only pointer generator model can output alignments to source sequence.");
                        }
                    }

                    using var g = computeGraph.CreateSubGraph($"TransformerDecoder_Alignment");
                    (var cost2, var bssSeqList) = Decoder.DecodeTransformer(tgtTokensList, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                                               originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType, 0.0f, decodingOptions, isTraining,
                                                                               outputSentScore: decodingOptions.BeamSearchSize > 1, pointerGenerator: pointerGenerator, 
                                                                               srcSeqs: srcTokensList, teacherForcedAlignment: true);
                    nr.Cost = 0.0f;
                    nr.Output = m_modelMetaData.TgtVocab.ExtractTokens(bssSeqList);
                    if (decodingOptions.OutputAligmentsToSrc)
                    {
                        (nr.Alignments, nr.AlignmentScores) = Decoder.ExtractAlignments(bssSeqList);
                    }
                }
                else
                {   // Test mode or running validation in Training mode
                    Dictionary<string, IWeightTensor> cachedTensors = new Dictionary<string, IWeightTensor>();
                    List<List<BeamSearchStatus>> beam2batchStatus = Decoder.InitBeamSearchStatusListList(batchSize, tgtTokensList);
                    for (int i = tgtTokensList[0].Count; i < decodingOptions.MaxTgtSentLength; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = null; //(batch_size, beam_search_size)
                        try
                        {
                            foreach (var batchStatus in beam2batchStatus)
                            {
                                var batch2tgtTokens = Decoder.ExtractBatchTokens(batchStatus);
                                List<List<int>> alignmentsToSrc = null;
                                List<List<float>> alignmentScores = null;
                                if (decodingOptions.OutputAligmentsToSrc)
                                {
                                    if (pointerGenerator == null)
                                    {
                                        throw new ArgumentException($"Only pointer generator model can output alignments to source sequence.");
                                    }

                                    (alignmentsToSrc, alignmentScores) = Decoder.ExtractBatchAlignments(batchStatus);
                                }

                                using var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}");
                                (var cost2, var bssSeqList) = Decoder.DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                                                originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType, 0.0f, decodingOptions, isTraining,
                                                                                outputSentScore: decodingOptions.BeamSearchSize > 1, previousBeamSearchResults: batchStatus,
                                                                                pointerGenerator: pointerGenerator, srcSeqs: srcTokensList,
                                                                                cachedTensors: cachedTensors, alignmentsToSrc: alignmentsToSrc, alignmentScoresToSrc: alignmentScores);

                                bssSeqList = Decoder.SwapBeamAndBatch(bssSeqList); // Swap shape: (beam_search_size, batch_size) -> (batch_size, beam_search_size)
                                batch2beam2seq = Decoder.CombineBeamSearchResults(batch2beam2seq, bssSeqList);
                            }
                        }
                        catch (OutOfMemoryException)
                        {
                            GC.Collect();
                            Logger.WriteLine(Logger.Level.warn, $"We have out of memory while generating '{i}th' tokens, so terminate decoding for current sequences.");
                            break;
                        }

                        if (decodingOptions.BeamSearchSize > 1)
                        {
                            // Keep top N result and drop all others
                            for (int k = 0; k < batchSize; k++)
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
                    nr.Output = m_modelMetaData.TgtVocab.ExtractTokens(beam2batchStatus);
                    if (decodingOptions.OutputAligmentsToSrc)
                    {
                        (nr.Alignments, nr.AlignmentScores) = Decoder.ExtractAlignments(beam2batchStatus);
                    }

                    if (cachedTensors != null)
                    {
                        foreach (var pair in cachedTensors)
                        {
                            pair.Value.Dispose();
                        }
                    }
                }
            }

            nr.RemoveDuplicatedEOS();

            nrs.Add(nr);
            return nrs;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_modelMetaData.SrcVocab.DumpVocab(outputSrcVocab);
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
        }
    }
}
