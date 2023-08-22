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
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using TensorSharp;

namespace Seq2SeqSharp.Applications
{
    public class GPT : BaseSeq2SeqFramework<Seq2SeqModel>
    {
        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding = null; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder = null; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer = null ; //The feed forward layers over devices after all layers in decoder
        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding = null;
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding = null;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        readonly Seq2SeqOptions m_options = null;

        private MemoryCache m_memoryCache;
        public GPT(Seq2SeqOptions options, Vocab tgtVocab = null)
            : base(deviceIds: options.DeviceIds, processorType: options.ProcessorType, modelFilePath: options.ModelFilePath, memoryUsageRatio: options.MemoryUsageRatio,
                  compilerOptions: options.CompilerOptions, runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq,
                  startToRunValidAfterUpdates: options.StartValidAfterUpdates, maxDegressOfParallelism: options.TaskParallelism, mklInstructions: options.MKLInstructions, weightsUpdateCount: options.WeightsUpdateCount, 
                  enableTensorCore: options.EnableTensorCore, cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32, randomSeed: options.RandomSeed)
        {
            m_shuffleType = options.ShuffleType;
            m_options = options;

            // Check if options are valided.
            m_options.ValidateOptions();

            m_memoryCache = new MemoryCache(new MemoryCacheOptions
            {
                SizeLimit = 1024
            });

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

            SaveModel(createBackupPrevious: true, suffix:".updatevocab");
        }

        public void VQModel()
        {
            m_modelMetaData.VQType = m_options.VQType;
            SaveModel(createBackupPrevious: true, suffix: $".{m_modelMetaData.VQType.ToString()}");

        }

        protected override Seq2SeqModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, Seq2SeqModel.Create, suffix);

        private bool CreateTrainableParameters(IModel model)
        {
            if (m_decoder != null)
            {
                m_decoder.Dispose();
            }
            if (m_decoderFFLayer != null)
            {
                m_decoderFFLayer.Dispose();
            }

            if (m_posEmbedding != null)
            {
                m_posEmbedding.Dispose();
            }

            if (m_segmentEmbedding != null)
            {
                m_segmentEmbedding.Dispose();
            }

            if (m_tgtEmbedding != null)
            {
                m_tgtEmbedding.Dispose();
            }

            Logger.WriteLine($"Creating decoders...");
            var raDeviceIds = new RoundArray<int>(DeviceIds);

            DType elementType = m_options.AMP ? DType.Float16 : DType.Float32;

            m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds, elementType);
            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: (m_options.Task == ModeEnums.Train), learningRateFactor: m_options.DecoderStartLearningRateFactor, elementType), DeviceIds);

            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, Math.Max(m_options.MaxTgtSentLength, m_options.MaxValidTgtSentLength), model, elementType, isTrainable: (m_options.Task == ModeEnums.Train));
            m_tgtEmbedding = CreateTgtEmbeddings(model, raDeviceIds, m_options.IsTgtEmbeddingTrainable && (m_options.Task == ModeEnums.Train), m_options.DecoderStartLearningRateFactor, elementType);

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
                    m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx),
                    m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx));
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
        /// 
        /// </summary>
        /// <param name="input">shape: [batch_size, seq_size]</param>
        /// <param name="output">shape: [beam_search_size, batch_size, seq_size]</param>
        /// <returns></returns>
        private List<List<List<string>>> CombineInputOutput(List<List<string>> input, List<List<List<string>>> output)
        {
            List<List<List<string>>> result = new List<List<List<string>>>();

            foreach (var batchSeqs in output)
            {
                List<List<string>> rBatchSeqs = new List<List<string>>();
                for (int i = 0;i < batchSeqs.Count;i++)
                {
                    List<string> r = new List<string>();
                    r.AddRange(input[i]);
                    r.AddRange(batchSeqs[i]);

                    rBatchSeqs.Add(r);
                }
                result.Add(rBatchSeqs);
            }

            return result;

        }


        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            (var decoder, var decoderFFLayer, var tgtEmbedding, var posEmbedding, var segmentEmbedding) = GetNetworksOnDeviceAt(computeGraph.DeviceId);
            List<NetworkResult> nrs = new List<NetworkResult>();

            // Generate output decoder sentences
            var tgtSnts = sntPairBatch.GetSrcTokens(0);
            int batchSize = tgtSnts.Count;
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);
            NetworkResult nr = new NetworkResult();
            nr.Status = NetworkResultStatus.SUCCEED;

            decoder.Reset(computeGraph.GetWeightFactory(), tgtSnts.Count);

            if (isTraining)
            {
                (var c, _) = Decoder.GPTDecode(tgtTokensList, computeGraph, decoder as GPTDecoder, decoderFFLayer, tgtEmbedding, posEmbedding, m_modelMetaData.TgtVocab, m_shuffleType,
                    m_options.DropoutRatio, decodingOptions, isTraining, lossType: m_options.LossType, focalLossGamma: m_options.FocalLossGamma, segmentEmbeddings: segmentEmbedding, amp: m_options.AMP);
                nr.Cost = c;
                nr.Output = null;
            }           
            else
            {   // Test mode or running validation in Training mode
                List<List<BeamSearchStatus>> beam2batchStatus = Decoder.InitBeamSearchStatusListList(batchSize, tgtTokensList);
                Dictionary<string, IWeightTensor> cachedTensors = null;
                string cacheKey = GenerateCacheKey(tgtSnts);
                if (!m_memoryCache.TryGetValue(cacheKey, out cachedTensors))
                {
                    cachedTensors = new Dictionary<string, IWeightTensor>();
                    if (Logger.Verbose == Logger.LogVerbose.Debug)
                    {
                        Logger.WriteLine($"Missed cached tensor. cacheKey = '{cacheKey}' key length = '{cacheKey.Length}'");
                    }
                }
                else
                {
                    if (Logger.Verbose == Logger.LogVerbose.Debug)
                    {
                        Logger.WriteLine($"Hit cached tensor. cacheKey = '{cacheKey}' key length = '{cacheKey.Length}'");
                    }
                }

                m_memoryCache.Remove(cacheKey);

                for (int i = tgtTokensList[0].Count; i < decodingOptions.MaxTgtSentLength; i++)
                {
                    List<List<BeamSearchStatus>> batch2beam2seq = null; //(batch_size, beam_search_size)
                    try
                    {
                        foreach (var batchStatus in beam2batchStatus)
                        {
                            var batch2tgtTokens = Decoder.ExtractBatchTokens(batchStatus);
                            using var g = computeGraph.CreateSubGraph($"GPTDecoder_Step_{i}");
                            (var cost2, var bssSeqList) = Decoder.GPTDecode(batch2tgtTokens, g, decoder as GPTDecoder, decoderFFLayer, tgtEmbedding, posEmbedding,
                                                                            m_modelMetaData.TgtVocab, m_shuffleType, 0.0f, decodingOptions, isTraining,
                                                                            outputSentScore: decodingOptions.BeamSearchSize > 1, previousBeamSearchResults: batchStatus,
                                                                            blockedTokens: decodingOptions.BlockedTokens, segmentEmbeddings: segmentEmbedding, cachedTensors: cachedTensors, amp: m_options.AMP);

                            bssSeqList = Decoder.SwapBeamAndBatch(bssSeqList); // Swap shape: (beam_search_size, batch_size) -> (batch_size, beam_search_size)
                            batch2beam2seq = Decoder.CombineBeamSearchResults(batch2beam2seq, bssSeqList);
                        }
                    }
                    catch (OutOfMemoryException)
                    {
                        GC.Collect();            
                        
                        // Release all items in cached tensors
                        if (cachedTensors!= null)
                        {
                            foreach (var pair in cachedTensors)
                            {
                                if (pair.Value != null)
                                {
                                    pair.Value.Dispose();
                                }
                            }
                            cachedTensors = null;
                        }

                        if (Logger.Verbose == Logger.LogVerbose.Debug)
                        {
                            Logger.WriteLine(Logger.Level.warn, $"We have out of memory while generating '{i}th' tokens, so terminate decoding for current sequences.");
                        }

                        nr.Status = NetworkResultStatus.OOM;
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

                Decoder.RemoveRange(beam2batchStatus, 0, tgtTokensList[0].Count);
                var generatedWords = m_modelMetaData.TgtVocab.CovertToWords(beam2batchStatus);
                nr.Output = CombineInputOutput(tgtSnts, generatedWords);

                if (cachedTensors != null)
                {
                    cacheKey = GenerateCacheKey(nr.Output[0]);
                    var cacheEntryOptions = new MemoryCacheEntryOptions().SetSize(1).SetAbsoluteExpiration(TimeSpan.FromMinutes(1));

                    Dictionary<string, IWeightTensor> newCachedTensors = new Dictionary<string, IWeightTensor>();
                    foreach (var pair in cachedTensors)
                    {
                        newCachedTensors.Add(pair.Key, pair.Value.CopyWeightsRef(pair.Value.Name, false, graphToBind: null));
                    }
                    m_memoryCache.Set(cacheKey, newCachedTensors, cacheEntryOptions);
                }
            }

            nr.RemoveDuplicatedEOS();

            nrs.Add(nr);
            return nrs;
        }

        public void DumpVocabToFiles(string outputTgtVocab)
        {
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
        }

        public void Test(string inputTestFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null)
        {
            Test<SeqCorpusBatch>(inputTestFile, outputFile, batchSize, decodingOptions, srcSpmPath, tgtSpmPath, outputAlignmentFile);
        }

        public void Test(string inputTestFile, string inputPromptFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null)
        {
            Test<SeqCorpusBatch>(inputTestFile, inputPromptFile, outputFile, batchSize, decodingOptions, srcSpmPath, tgtSpmPath, outputAlignmentFile);
        }
    }
}
