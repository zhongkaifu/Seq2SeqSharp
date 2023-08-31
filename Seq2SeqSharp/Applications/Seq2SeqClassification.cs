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
using Seq2SeqSharp.Enums;
using TensorSharp;

namespace Seq2SeqSharp.Applications
{
    public class Seq2SeqClassification : BaseSeq2SeqFramework<Seq2SeqClassificationModel>
    {
        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source

        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices.
        private MultiProcessorNetworkWrapper<IDecoder> m_decoder; //The decoders over devices
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_encoderFFLayer; //The feed forward layers over devices after all layers in encoder
        private MultiProcessorNetworkWrapper<IFeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after all layers in decoder

        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;

        private readonly ShuffleEnums m_shuffleType = ShuffleEnums.Random;
        private readonly Seq2SeqClassificationOptions m_options;

        public Vocab ClsVocab => m_modelMetaData.ClsVocab;

        public Seq2SeqClassification(Seq2SeqClassificationOptions options, Vocab srcVocab = null, Vocab tgtVocab = null, Vocab clsVocab = null)
            : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, 
                  runValidEveryUpdates: options.RunValidEveryUpdates, primaryTaskId: options.PrimaryTaskId, updateFreq: options.UpdateFreq, 
                  startToRunValidAfterUpdates: options.StartValidAfterUpdates, maxDegressOfParallelism: options.TaskParallelism, 
                  cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32)
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
                if (srcVocab != null || tgtVocab != null || clsVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                m_modelMetaData = LoadModel();
            }
            else
            {
                m_modelMetaData = new Seq2SeqClassificationModel(options, srcVocab, tgtVocab, clsVocab);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();
        }

        protected override Seq2SeqClassificationModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, Seq2SeqClassificationModel.Create, suffix);
        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            var raDeviceIds = new RoundArray<int>(DeviceIds);
            m_encoder = Encoder.CreateEncoders(model, m_options, raDeviceIds);
            m_decoder = Decoder.CreateDecoders(model, m_options, raDeviceIds);

            m_encoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Encoder_0", model.HiddenDim, model.ClsVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true), DeviceIds);

            m_decoderFFLayer = new MultiProcessorNetworkWrapper<IFeedForwardLayer>(new FeedForwardLayer("FeedForward_Decoder_0", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(),
                isTrainable: true), DeviceIds);

            m_segmentEmbedding = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, Math.Max(Math.Max(m_options.MaxSrcSentLength, m_options.MaxValidSrcSentLength), Math.Max(m_options.MaxTgtSentLength, m_options.MaxValidTgtSentLength)), model);
            (m_srcEmbedding, m_tgtEmbedding) = CreateSrcTgtEmbeddings(model, raDeviceIds, m_options.IsSrcEmbeddingTrainable, m_options.IsTgtEmbeddingTrainable, m_options.EncoderStartLearningRateFactor, m_options.DecoderStartLearningRateFactor);
            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, IDecoder, IFeedForwardLayer, IFeedForwardLayer, IWeightTensor, IWeightTensor, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx),
                    m_decoder.GetNetworkOnDevice(deviceIdIdx),
                    m_encoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx),
                    m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_modelMetaData.SharedEmbeddings ? m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx) : m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx),
                    m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx));
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
            (IEncoder encoder, IDecoder decoder, IFeedForwardLayer encoderFFLayer, IFeedForwardLayer decoderFFLayer, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, IWeightTensor segmentEmbedding) = GetNetworksOnDeviceAt(computeGraph.DeviceId);

            var srcSnts = sntPairBatch.GetSrcTokens(0);
            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);

            IWeightTensor encOutput = Encoder.Run(computeGraph, sntPairBatch, encoder, m_modelMetaData, m_shuffleType, srcEmbedding, segmentEmbedding, srcTokensList, originalSrcLengths);

            List<NetworkResult> nrs = new List<NetworkResult>();
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

            float cost = 0.0f;
            NetworkResult nrCLS = new NetworkResult
            {
                Output = new List<List<List<string>>>()
            };

            IWeightTensor ffLayer = encoderFFLayer.Process(clsWeightTensor, batchSize, computeGraph);
            using (IWeightTensor probs = computeGraph.Softmax(ffLayer, runGradients: false, inPlace: true))
            {
                if (isTraining)
                {
                    var clsSnts = sntPairBatch.GetTgtTokens(0);
                    for (int k = 0; k < batchSize; k++)
                    {
                        int ix_targets_k_j = m_modelMetaData.ClsVocab.GetWordIndex(clsSnts[k][0]);
                        float score_k = probs.GetWeightAt(new long[] { k, ix_targets_k_j });
                        cost += (float)-Math.Log(score_k);
                        probs.SetWeightAt(score_k - 1, new long[] { k, ix_targets_k_j });
                    }

                    ffLayer.CopyWeightsToGradients(probs);

                    nrCLS.Cost = cost / batchSize;
                }
                else
                {
                    // Output "i"th target word
                    using var targetIdxTensor = computeGraph.Argmax(probs, 1);
                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    List<string> targetWords = m_modelMetaData.ClsVocab.ConvertIdsToString(targetIdx.ToList());
                    nrCLS.Output.Add(new List<List<string>>());

                    for (int k = 0; k < batchSize; k++)
                    {
                        nrCLS.Output[0].Add(new List<string>());
                        nrCLS.Output[0][k].Add(targetWords[k]);
                    }
                }
            }

            // Reset networks
            decoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);

            // Generate output decoder sentences
            var tgtSnts = sntPairBatch.GetTgtTokens(1);
            var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);

            NetworkResult nr = new NetworkResult();
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
                    (var c, _) = Decoder.DecodeTransformer(tgtTokensList, computeGraph, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding, originalSrcLengths, m_modelMetaData.TgtVocab, 
                        m_shuffleType, m_options.DropoutRatio, decodingOptions, isTraining, segmentEmbeddings: segmentEmbedding);
                    nr.Cost = c;
                    nr.Output = null;
                }
                else
                {
                    List<List<BeamSearchStatus>> beam2batchStatus = Decoder.InitBeamSearchStatusListList(batchSize, tgtTokensList);
                    for (int i = 0; i < decodingOptions.MaxTgtSentLength; i++)
                    {
                        List<List<BeamSearchStatus>> batch2beam2seq = null; //(batch_size, beam_search_size)
                        try
                        {
                            foreach (var batchStatus in beam2batchStatus)
                            {
                                var batch2tgtTokens = Decoder.ExtractBatchTokens(batchStatus);
                                using var g = computeGraph.CreateSubGraph($"TransformerDecoder_Step_{i}");
                                (var cost2, var bssSeqList) = Decoder.DecodeTransformer(batch2tgtTokens, g, encOutput, decoder as TransformerDecoder, decoderFFLayer, tgtEmbedding,
                                                                                originalSrcLengths, m_modelMetaData.TgtVocab, m_shuffleType, 0.0f, decodingOptions, isTraining,
                                                                                outputSentScore: decodingOptions.BeamSearchSize > 1, previousBeamSearchResults: batchStatus, segmentEmbeddings: segmentEmbedding);

                                bssSeqList = Decoder.SwapBeamAndBatch(bssSeqList);
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
                    nr.Output = m_modelMetaData.TgtVocab.CovertToWords(beam2batchStatus);
                }
            }

            nr.RemoveDuplicatedEOS();

            nrs.Add(nrCLS);
            nrs.Add(nr);

            return nrs;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab, string outputClsVocab)
        {
            m_modelMetaData.SrcVocab.DumpVocab(outputSrcVocab);
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
            m_modelMetaData.ClsVocab.DumpVocab(outputClsVocab);
        }
    }
}
