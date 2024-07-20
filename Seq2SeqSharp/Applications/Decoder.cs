// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using TensorSharp;
using Seq2SeqSharp.Enums;
using ProtoBuf;

namespace Seq2SeqSharp.Applications
{
    public class Decoder
    {
        public static MultiProcessorNetworkWrapper<IDecoder> CreateDecoders(IModel model, Seq2SeqOptions options, RoundArray<int> raDeviceIds, DType elementType = DType.Float32)
        {
            MultiProcessorNetworkWrapper<IDecoder> decoder;
            if (model.DecoderType == DecoderTypeEnums.AttentionLSTM)
            {
                decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                     new AttentionDecoder("AttnLSTMDecoder", model.HiddenDim, model.DecoderEmbeddingDim, model.HiddenDim,
                     options.DropoutRatio, model.DecoderLayerDepth, raDeviceIds.GetNextItem(), model.EnableCoverageModel, 
                     isTrainable: options.IsDecoderTrainable && (options.Task == ModeEnums.Train), elementType: elementType), raDeviceIds.ToArray());
            }
            else if (model.DecoderType == DecoderTypeEnums.GPTDecoder)
            {
                decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new GPTDecoder("GPTDecoder", model.MultiHeadNum, model.HiddenDim, model.IntermediateDim, model.DecoderEmbeddingDim, model.DecoderLayerDepth, options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: options.IsDecoderTrainable && (options.Task == ModeEnums.Train), learningRateFactor: options.DecoderStartLearningRateFactor, activateFunc: model.ActivateFunc, expertNum: model.ExpertNum, 
                    expertsPerTokenFactor: model.ExpertsPerTokenFactor, elementType: elementType, peType:model.PEType, normType: model.NormType, attentionType: options.AttentionType), raDeviceIds.ToArray());
            }
            else
            {
                decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new TransformerDecoder("TransformerDecoder", model.MultiHeadNum, model.HiddenDim, model.IntermediateDim, model.DecoderEmbeddingDim, model.DecoderLayerDepth, options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: options.IsDecoderTrainable && (options.Task == ModeEnums.Train), learningRateFactor: options.DecoderStartLearningRateFactor, activateFunc: model.ActivateFunc, expertNum: model.ExpertNum, 
                    expertsPerTokenFactor: model.ExpertsPerTokenFactor, elementType: elementType, peType:model.PEType, normType: model.NormType, attentionType: options.AttentionType), raDeviceIds.ToArray());
            }

            return decoder;
        }


        /// <summary>
        /// Extract tokens from given batch status
        /// Input shape: (batch_size)
        /// </summary>
        /// <param name="batchStatus"></param>
        /// <returns>shape: [batch_size, seq_len]</returns>
        public static List<List<int>> ExtractBatchTokens(List<BeamSearchStatus> batchStatus, int index = -1, int count = -1)
        {
            List<List<int>> batchTokens = new List<List<int>>();            
            foreach (var item in batchStatus)
            {
                if (index < 0 || count < 0)
                {
                    batchTokens.Add(item.OutputIds);
                }
                else
                {
                    batchTokens.Add(item.OutputIds.GetRange(index, count));
                }
            }

            return batchTokens;
        }


        /// <summary>
        /// Extract aligments from given batch status
        /// Input shape: (batch_size)
        /// </summary>
        /// <param name="batchStatus"></param>
        /// <returns>shape: [batch_size, seq_len]</returns>
        public static (List<List<int>>, List<List<float>>) ExtractBatchAlignments(List<BeamSearchStatus> batchStatus, int index = -1, int count = -1)
        {
            List<List<int>> batchAlignments = new List<List<int>>();
            List<List<float>> batchAlignmentScores = new List<List<float>>();

            foreach (var item in batchStatus)
            {
                if (index < 0 || count < 0)
                {
                    batchAlignments.Add(item.AlignmentsToSrc);
                    batchAlignmentScores.Add(item.AlignmentScores);
                }
                else
                {
                    batchAlignments.Add(item.AlignmentsToSrc.GetRange(index, count));
                    batchAlignmentScores.Add(item.AlignmentScores.GetRange(index, count));
                }
            }

            return (batchAlignments, batchAlignmentScores);
        }

        /// <summary>
        /// Initialize beam search status for all beam search and batch results
        /// </summary>
        /// <param name="beamSearchSize"></param>
        /// <param name="batchSize"></param>
        /// <param name="tgtTokensList"></param>
        /// <returns>shape: (beam_search_size, batch_size)</returns>
        public static List<List<BeamSearchStatus>> InitBeamSearchStatusListList(int batchSize, List<List<int>> tgtTokensList)
        {
            List<List<BeamSearchStatus>> beam2batchStatus = new List<List<BeamSearchStatus>>();

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();
            for (int i = 0; i < batchSize; i++)
            {
                BeamSearchStatus bss = new BeamSearchStatus();
                bss.OutputIds.AddRange(tgtTokensList[i]);

                for (int j = 0; j < bss.OutputIds.Count; j++)
                {
                    bss.AlignmentsToSrc.Add(-1);
                    bss.AlignmentScores.Add(0.0f);
                }

                bssList.Add(bss);
            }

            beam2batchStatus.Add(bssList);

            return beam2batchStatus;
        }


        /// <summary>
        /// Extract alginments from beam search result
        /// </summary>
        /// <param name="beam2batch2seq"></param>
        /// <returns></returns>
        public static (List<List<List<int>>>, List<List<List<float>>>) ExtractAlignments(List<List<BeamSearchStatus>> beam2batch2seq)
        {
            List<List<List<int>>> result = new List<List<List<int>>>();
            List<List<List<float>>> scores = new List<List<List<float>>>();
            foreach (var batch2seq in beam2batch2seq)
            {
                List<List<int>> b = new List<List<int>>();
                List<List<float>> bScores = new List<List<float>>();
                foreach (var seq in batch2seq)
                {
                    List<int> r = new List<int>();
                    foreach (int idx in seq.AlignmentsToSrc)
                    {
                        r.Add(idx);
                    }
                    b.Add(r);

                    List<float> rScores = new List<float>();
                    foreach (float score in seq.AlignmentScores)
                    {
                        rScores.Add(score);
                    }
                    bScores.Add(rScores);
                }
                result.Add(b);
                scores.Add(bScores);

            }

            return (result, scores);
        }


        /// <summary>
        /// swap shape: (beam_search_size, batch_size) <-> (batch_size, beam_search_size)
        /// </summary>
        /// <param name="input"></param>
        public static List<List<BeamSearchStatus>> SwapBeamAndBatch(List<List<BeamSearchStatus>> input)
        {
            int size1 = input.Count;
            int size2 = input[0].Count;

            List<List<BeamSearchStatus>> output = new List<List<BeamSearchStatus>>();
            for (int k = 0; k < size2; k++)
            {
                output.Add(new List<BeamSearchStatus>());
            }

            for (int j = 0; j < size1; j++)
            {
                for (int k = 0; k < size2; k++)
                {
                    output[k].Add(input[j][k]);
                }
            }

            return output;
        }

        public static bool AreAllSentsCompleted(List<List<BeamSearchStatus>> input)
        {
            foreach (var seqs in input)
            {
                foreach (var seq in seqs)
                {
                    if (seq.OutputIds[^1] != (int)SENTTAGS.END)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Remove tokens range
        /// </summary>
        /// <param name="beam2batch2seq">Shape: [Beam_Search_Size, Batch_Size, Sequence_Length]</param>
        /// <returns></returns>
        public static void RemoveRange(List<List<BeamSearchStatus>> beam2batch2seq, int idx, int count)
        {
            foreach (var batch2seq in beam2batch2seq)
            {
                foreach (var seq in batch2seq)
                {
                    seq.OutputIds.RemoveRange(idx, count);
                }
            }
        }


        /// <summary>
        /// Combine two beam search results to a single result
        /// Input shape: (batch_size, beam_search_size_1) and (batch_size, beam_search_size_2)
        /// Output shape: (batch_size, beam_search_size_1 + beam_search_size_2)
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <returns></returns>
        public static List<List<BeamSearchStatus>> CombineBeamSearchResults(List<List<BeamSearchStatus>> input1, List<List<BeamSearchStatus>> input2)
        {
            if (input1 == null)
            {
                return input2;
            }

            if (input2 == null)
            {
                return input1;
            }

            List<List<BeamSearchStatus>> output = new List<List<BeamSearchStatus>>();
            for (int i = 0; i < input1.Count; i++)
            {
                output.Add(new List<BeamSearchStatus>());
                output[i].AddRange(input1[i]);
                output[i].AddRange(input2[i]);
            }

            return output;
        }

        public static (float, List<List<BeamSearchStatus>>) DecodeTransformer(List<List<int>> tgtSeqs, IComputeGraph g, IWeightTensor encOutputs, TransformerDecoder decoder, IFeedForwardLayer decoderFFLayer,
            IWeightTensor tgtEmbedding, float[] srcOriginalLenghts, Vocab tgtVocab, PaddingEnums paddingType, float dropoutRatio, DecodingOptions decodingOptions, bool isTraining = true,
            bool outputSentScore = true, List<BeamSearchStatus> previousBeamSearchResults = null, IFeedForwardLayer pointerGenerator = null, List<List<int>> srcSeqs = null, Dictionary<string, IWeightTensor> cachedTensors = null,
            List<List<int>> alignmentsToSrc = null, List<List<float>> alignmentScoresToSrc = null, bool teacherForcedAlignment = false, LossEnums lossType = LossEnums.CrossEntropy, float focalLossGamma = 0.0f, float lossSmooth = 1e-9f, 
            List<int> blockedTokens = null, IWeightTensor segmentEmbeddings = null, bool amp = false, IWeightTensor posEmbeddings = null, float lossScaling = 1.0f)
        {
            int eosTokenId = tgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            int batchSize = tgtSeqs.Count;
            var tgtOriginalLengths = BuildInTokens.PadSentences(tgtSeqs, eosTokenId);
            int tgtSeqLen = tgtSeqs[0].Count;
            int srcSeqLen = encOutputs.Rows / batchSize;
            IWeightTensor srcTgtMask = (paddingType == PaddingEnums.NoPadding || batchSize == 1) ? null : g.BuildSrcTgtMask(srcSeqLen, tgtSeqLen, tgtOriginalLengths, srcOriginalLenghts, amp ? TensorSharp.DType.Float16 : TensorSharp.DType.Float32);
            if (srcTgtMask != null)
            {
                srcTgtMask = g.View(srcTgtMask, new long[] { srcTgtMask.Sizes[0], 1, srcTgtMask.Sizes[1], srcTgtMask.Sizes[2] });
            }

            IWeightTensor tgtSelfTriMask;
            if (paddingType == PaddingEnums.NoPadding || paddingType == PaddingEnums.NoPaddingInTgt || batchSize == 1)
            {
                tgtSelfTriMask = g.BuildTriMask(tgtSeqLen, batchSize, amp ? TensorSharp.DType.Float16 : TensorSharp.DType.Float32);
                tgtSelfTriMask = g.View(tgtSelfTriMask, new long[] { 1, 1, tgtSeqLen, tgtSeqLen });
            }
            else
            {
                tgtSelfTriMask = g.BuildSelfTriMask(tgtSeqLen, tgtOriginalLengths, amp ? TensorSharp.DType.Float16 : TensorSharp.DType.Float32);
                tgtSelfTriMask = g.View(tgtSelfTriMask, new long[] { batchSize, 1, tgtSeqLen, tgtSeqLen });
            }

            IWeightTensor inputEmbs = TensorUtils.CreateTokensEmbeddings(tgtSeqs, g, tgtEmbedding, segmentEmbeddings, tgtVocab, scaleFactor: (float)Math.Sqrt(tgtEmbedding.Columns), amp: amp);
            if (posEmbeddings != null)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, dropoutRatio);
            }

            IWeightTensor decOutput;
            IWeightTensor decEncAttnProbs;           
            (decOutput, decEncAttnProbs) = decoder.Decode(inputEmbs, encOutputs, tgtSelfTriMask, srcTgtMask, batchSize, g, outputAttnWeights: pointerGenerator != null, cachedTensors: cachedTensors);

            if (isTraining == false && teacherForcedAlignment == false)
            {
                // For inference, we only process last token of each sequence in order to speed up
                float[] decOutputIdx = new float[batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    decOutputIdx[i] = tgtSeqLen * (i + 1) - 1;
                }


                var indice = g.CreateTensorWeights(new long[] { decOutputIdx.Length, 1 }, decOutputIdx, dtype: DType.Float32);
                decOutput = g.IndexSelect(decOutput, indice);
                if (pointerGenerator != null)
                {
                    decEncAttnProbs = g.IndexSelect(decEncAttnProbs, indice);
                }

                tgtSeqLen = 1;
            }

            IWeightTensor ffLayer = decoderFFLayer.Process(decOutput, batchSize, g);

            if (isTraining == false && decodingOptions.DecodingStrategy == DecodingStrategyEnums.Sampling && decodingOptions.Temperature != 1.0f)
            {
                ffLayer = g.Div(ffLayer, decodingOptions.Temperature, inPlace: true);
            }

            if (amp)
            {
                var tmp = ffLayer;
                ffLayer = g.Half2Float(ffLayer);
                tmp.ReleaseWeight();
            }

            IWeightTensor probs = (lossType == LossEnums.NegativeLogLikelihood && isTraining) ? g.LogSoftmax(ffLayer) : g.Softmax(ffLayer, inPlace: true);
            IWeightTensor probsCopy = null;
            if (pointerGenerator != null)
            {
                if (amp)
                {
                    decEncAttnProbs = g.Half2Float(decEncAttnProbs);
                    encOutputs = g.Half2Float(encOutputs);
                }

                //Build onehot tensor for source tokens
                var seqSeqsIndex = g.CreateTensorForIndex(srcSeqs);
                seqSeqsIndex = g.View(seqSeqsIndex, dims: new long[] { batchSize, 1, srcSeqLen });
                seqSeqsIndex = g.AsContiguous(g.Expand(seqSeqsIndex, dims: new long[] { batchSize, tgtSeqLen, srcSeqLen }));
                seqSeqsIndex = g.View(seqSeqsIndex, dims: new long[] { batchSize * tgtSeqLen, srcSeqLen });

                //Build context tensor for pointer generator
                IWeightTensor decEncAttnProbsBatch = g.View(decEncAttnProbs, dims: new long[] { batchSize, tgtSeqLen, srcSeqLen });
                IWeightTensor encOutputsBatch = g.View(encOutputs, dims: new long[] { batchSize, srcSeqLen, -1 });

                var pointer_context = g.MulBatch(decEncAttnProbsBatch, encOutputsBatch); //Output: [batchSize, tgtSeqLen, embedding_size]
                pointer_context = g.View(pointer_context, dims: new long[] { batchSize * tgtSeqLen, -1 });

                var p_copy = pointerGenerator.Process(pointer_context, batchSize, g); // Output: [batchSize * tgtSeqLen, 1]
                p_copy = g.Sigmoid(p_copy);

                var p_gen = g.Sub(1.0f, p_copy);
                p_gen = g.Expand(p_gen, dims: new long[] { batchSize * tgtSeqLen, ffLayer.Sizes[^1] });

                //Apply copy probs to attention weights in source side
                p_copy = g.Expand(p_copy, dims: new long[] { batchSize * tgtSeqLen, srcSeqLen });
                probsCopy = g.EltMul(p_copy, decEncAttnProbs); // Output shape: [batchSize * tgtSeqLen, srcSeqLen]

                var probsCopyScatter = g.ScatterAdd(probsCopy, seqSeqsIndex, 1, shape: new long[] { batchSize * tgtSeqLen, ffLayer.Sizes[^1] });

                if (lossType == LossEnums.NegativeLogLikelihood && isTraining)
                {
                    probs = g.Exp(probs);
                }

                probs = g.EltMul(probs, p_gen);
                probs = g.Add(probs, probsCopyScatter, inPlace: true);

                if (lossType == LossEnums.NegativeLogLikelihood && isTraining)
                {
                    probs = g.Log(probs);
                }

            }

            if (isTraining)
            {
                var leftShiftTgtSeqs = g.LeftShiftTokens(tgtSeqs, eosTokenId);
                var cost = lossType == LossEnums.CrossEntropy ? g.CrossEntropyLoss(probs, leftShiftTgtSeqs, graident: lossScaling, smooth: lossSmooth, gamma: focalLossGamma) : g.NLLLoss(probs, leftShiftTgtSeqs);

                return (cost, null);
            }
            else
            {
                if (decodingOptions.BlockedTokens != null && decodingOptions.BlockedTokens.Count > 0)
                {
                    var btList = new List<List<int>>();
                    btList.Add(decodingOptions.BlockedTokens);
                    var blockTokensIdxTensor = g.CreateTensorForIndex(btList); // [1, BlockedTokens.Count]
                    var blockTokensTensor = g.Scatter(blockTokensIdxTensor, -1.0f, 1, probs.ElementType, false, shape: new long[] { 1, probs.Sizes[1] });
                    blockTokensTensor = g.Expand(blockTokensTensor, dims: probs.Sizes);
                    probs = g.Add(blockTokensTensor, probs);
                }

                // Transformer decoder with beam search at inference time
                List<List<BeamSearchStatus>> bssSeqList = new List<List<BeamSearchStatus>>(); //shape: (beam_search_size, batch_size)
                int beamSearchSize = decodingOptions.BeamSearchSize;
                while (beamSearchSize > 0)
                {
                    // Output "i"th target word
                    using var targetIdxTensor = (decodingOptions.DecodingStrategy == DecodingStrategyEnums.GreedySearch) ? g.Argmax(probs, 1) : 
                                                g.TopPSample(probs, decodingOptions.TopP, decodingOptions.RepeatPenalty, decodedSequences: tgtSeqs);
                    IWeightTensor gatherTensor = null;
                    if (outputSentScore)
                    {
                        gatherTensor = g.Gather(probs, targetIdxTensor, 1);
                        gatherTensor = g.Log(gatherTensor);
                    }

                    float[] alignmentsIdx = null;
                    float[] alignmentScores = null;
                    if (alignmentsToSrc != null || teacherForcedAlignment)
                    {
                        using var alignmentsIdxTensor = g.Argmax(decEncAttnProbs, 1);
                        alignmentsIdx = alignmentsIdxTensor.ToWeightArray();

                        
                        using var alignmentScoresIdxTensor = g.Gather(probsCopy, alignmentsIdxTensor, 1);
                        alignmentScores = alignmentScoresIdxTensor.ToWeightArray();
                    }

                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    List<BeamSearchStatus> outputTgtSeqs = new List<BeamSearchStatus>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        BeamSearchStatus bss = new BeamSearchStatus();
                        if (teacherForcedAlignment)
                        {
                            for (int j = 0; j < tgtSeqLen; j++)
                            {
                                bss.OutputIds.Add((int)targetIdx[i * tgtSeqLen + j]);
                                bss.AlignmentsToSrc.Add((int)(alignmentsIdx[i * tgtSeqLen + j]));
                                bss.AlignmentScores.Add(alignmentScores[i * tgtSeqLen + j]);
                            }
                        }
                        else
                        {
                            bss.OutputIds.AddRange(tgtSeqs[i]);
                            bss.OutputIds.Add((int)(targetIdx[i]));

                            if (alignmentsIdx != null)
                            {
                                bss.AlignmentsToSrc.AddRange(alignmentsToSrc[i]);
                                bss.AlignmentsToSrc.Add((int)(alignmentsIdx[i]));

                                bss.AlignmentScores.AddRange(alignmentScoresToSrc[i]);
                                bss.AlignmentScores.Add(alignmentScores[i]);
                            }

                            if (outputSentScore)
                            {
                                bss.Score = previousBeamSearchResults[i].Score + -1.0f * gatherTensor.GetWeightAt(new long[] { i, 0 });
                            }
                        }



                        outputTgtSeqs.Add(bss);
                    }

                    bssSeqList.Add(outputTgtSeqs);

                    beamSearchSize--;
                    if (beamSearchSize > 0)
                    {
                        TensorUtils.ScatterFill(probs, 0.0f, targetIdxTensor, 1);
                    }
                }
                return (0.0f, bssSeqList);
            }
        }


        public static (float, List<List<BeamSearchStatus>>) GPTDecode(List<List<int>> tgtSeqs, IComputeGraph g, GPTDecoder decoder, IFeedForwardLayer decoderFFLayer,
            IWeightTensor tgtEmbedding, Vocab tgtVocab, PaddingEnums paddingType, float dropoutRatio, DecodingOptions decodingOptions, bool isTraining = true,
            bool outputSentScore = true, List<BeamSearchStatus> previousBeamSearchResults = null, Dictionary<string, IWeightTensor> cachedTensors = null,
            LossEnums lossType = LossEnums.CrossEntropy, float focalLossGamma = 0.0f, float lossSmooth = 1e-9f, IWeightTensor segmentEmbeddings = null, bool amp = true,
            IWeightTensor posEmbeddings = null, float lossScaling = 1.0f)
        {
            int eosTokenId = tgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            int batchSize = tgtSeqs.Count;
            var tgtOriginalLengths = BuildInTokens.PadSentences(tgtSeqs, eosTokenId);
            int tgtSeqLen = tgtSeqs[0].Count;

            IWeightTensor tgtSelfTriMask;
            if (paddingType == PaddingEnums.NoPadding || paddingType == PaddingEnums.NoPaddingInTgt || batchSize == 1)
            {
                tgtSelfTriMask = g.BuildTriMask(tgtSeqLen, batchSize, amp ? TensorSharp.DType.Float16 : TensorSharp.DType.Float32);
                tgtSelfTriMask = g.View(tgtSelfTriMask, new long[] { 1, 1, tgtSeqLen, tgtSeqLen });
            }
            else
            {
                tgtSelfTriMask = g.BuildSelfTriMask(tgtSeqLen, tgtOriginalLengths, amp ? TensorSharp.DType.Float16 : TensorSharp.DType.Float32);
                tgtSelfTriMask = g.View(tgtSelfTriMask, new long[] { batchSize, 1, tgtSeqLen, tgtSeqLen });
            }

            IWeightTensor inputEmbs = TensorUtils.CreateTokensEmbeddings(tgtSeqs, g, tgtEmbedding, segmentEmbeddings, tgtVocab, scaleFactor: (float)Math.Sqrt(tgtEmbedding.Columns), amp: amp);
            if (posEmbeddings != null)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, dropoutRatio);
            }

            IWeightTensor decOutput;
            (decOutput, _) = decoder.Decode(inputEmbs, tgtSelfTriMask, batchSize, g, cachedTensors: cachedTensors);

            if (isTraining == false)
            {
                // For inference, we only process last token of each sequence in order to speed up
                float[] decOutputIdx = new float[batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    decOutputIdx[i] = tgtSeqLen * (i + 1) - 1;
                }

                var indice = g.CreateTensorWeights(new long[] { decOutputIdx.Length, 1 }, decOutputIdx, dtype: DType.Float32);
                decOutput = g.IndexSelect(decOutput, indice);
            }

            IWeightTensor ffLayer = decoderFFLayer.Process(decOutput, batchSize, g);

            if (isTraining == false && decodingOptions.DecodingStrategy == DecodingStrategyEnums.Sampling && decodingOptions.Temperature != 1.0f)
            {
                ffLayer = g.Div(ffLayer, decodingOptions.Temperature, inPlace: true);
            }

            if (amp)
            {
                var tmp = ffLayer;
                ffLayer = g.Half2Float(ffLayer);
                tmp.ReleaseWeight();
            }

            IWeightTensor probs = (lossType == LossEnums.NegativeLogLikelihood && isTraining) ? g.LogSoftmax(ffLayer) : g.Softmax(ffLayer, inPlace: true);


            if (isTraining)
            {
                var leftShiftTgtSeqs = g.LeftShiftTokens(tgtSeqs, eosTokenId);
                var cost = lossType == LossEnums.CrossEntropy ? g.CrossEntropyLoss(probs, leftShiftTgtSeqs, graident: lossScaling, smooth: lossSmooth, gamma: focalLossGamma) : g.NLLLoss(probs, leftShiftTgtSeqs);

                return (cost, null);
            }
            else
            {
                if (decodingOptions.BlockedTokens != null && decodingOptions.BlockedTokens.Count > 0)
                {
                    var btList = new List<List<int>>();
                    btList.Add(decodingOptions.BlockedTokens);
                    var blockTokensIdxTensor = g.CreateTensorForIndex(btList); // [1, BlockedTokens.Count]
                    var blockTokensTensor = g.Scatter(blockTokensIdxTensor, -1.0f, 1, probs.ElementType, false, shape: new long[] { 1, probs.Sizes[1] });
                    blockTokensTensor = g.Expand(blockTokensTensor, dims: probs.Sizes);
                    probs = g.Add(blockTokensTensor, probs);
                }

                // Transformer decoder with beam search at inference time
                List<List<BeamSearchStatus>> bssSeqList = new List<List<BeamSearchStatus>>(); //shape: (beam_search_size, batch_size)
                int beamSearchSize = decodingOptions.BeamSearchSize;
                while (beamSearchSize > 0)
                {
                    // Output "i"th target word
                    using var targetIdxTensor = (decodingOptions.DecodingStrategy == DecodingStrategyEnums.GreedySearch) ? g.Argmax(probs, 1) :
                                                g.TopPSample(probs, decodingOptions.TopP, decodingOptions.RepeatPenalty, decodedSequences: tgtSeqs);
                    IWeightTensor gatherTensor = null;
                    if (outputSentScore)
                    {
                        gatherTensor = g.Gather(probs, targetIdxTensor, 1);
                        gatherTensor = g.Log(gatherTensor);
                    }

                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    List<BeamSearchStatus> outputTgtSeqs = new List<BeamSearchStatus>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        BeamSearchStatus bss = new BeamSearchStatus();

                        bss.OutputIds.AddRange(tgtSeqs[i]);
                        bss.OutputIds.Add((int)(targetIdx[i]));
                        if (outputSentScore)
                        {
                            bss.Score = previousBeamSearchResults[i].Score + -1.0f * gatherTensor.GetWeightAt(new long[] { i, 0 });
                        }
                        outputTgtSeqs.Add(bss);
                    }

                    bssSeqList.Add(outputTgtSeqs);

                    beamSearchSize--;
                    if (beamSearchSize > 0)
                    {
                        TensorUtils.ScatterFill(probs, 0.0f, targetIdxTensor, 1);
                    }
                }
                return (0.0f, bssSeqList);
            }
        }

        /// <summary>
        /// Decode output sentences in training
        /// </summary>
        /// <param name="outputSnts">In training mode, they are golden target sentences, otherwise, they are target sentences generated by the decoder</param>
        /// <param name="g"></param>
        /// <param name="encOutputs"></param>
        /// <param name="decoder"></param>
        /// <param name="decoderFFLayer"></param>
        /// <param name="tgtEmbedding"></param>
        /// <returns></returns>
        static public float DecodeAttentionLSTM(List<List<int>> outputSnts, IComputeGraph g, IWeightTensor encOutputs, AttentionDecoder decoder, IFeedForwardLayer decoderFFLayer, IWeightTensor tgtEmbedding, Vocab tgtVocab, int batchSize, bool isTraining = true)
        {
            int eosTokenId = tgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            float cost = 0.0f;
            float[] ix_inputs = new float[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = outputSnts[i][0];
            }

            // Initialize variables accoridng to current mode
            var originalOutputLengths = isTraining ? BuildInTokens.PadSentences(outputSnts, eosTokenId) : null;
            int seqLen = isTraining ? outputSnts[0].Count : 64;
            HashSet<int> setEndSentId = isTraining ? null : new HashSet<int>();

            // Pre-process for attention model
            AttentionPreProcessResult attPreProcessResult = decoder.PreProcess(encOutputs, batchSize, g);
            for (int i = 1; i < seqLen; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    inputs.Add(g.Peek(tgtEmbedding, 0, (int)ix_inputs[j]));
                }
                IWeightTensor inputsM = g.Concate(inputs, 0);

                //Decode output sentence at position i
                IWeightTensor dOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);
                IWeightTensor ffLayer = decoderFFLayer.Process(dOutput, batchSize, g);

                //Softmax for output
                using IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true);
                if (isTraining)
                {
                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        float score_k = probs.GetWeightAt(new long[] { k, outputSnts[k][i] });
                        if (i < originalOutputLengths[k])
                        {
                            var lcost = (float)-Math.Log(score_k);
                            if (float.IsNaN(lcost))
                            {
                                throw new ArithmeticException($"Score = '{score_k}' Cost = Nan at index '{i}' word '{outputSnts[k][i]}', Output Sentence = '{string.Join(" ", outputSnts[k])}'");
                            }
                            else
                            {
                                cost += lcost;
                            }
                        }

                        probs.SetWeightAt(score_k - 1, new long[] { k, outputSnts[k][i] });
                        ix_inputs[k] = outputSnts[k][i];
                    }
                    ffLayer.CopyWeightsToGradients(probs);
                }
                else
                {
                    // Output "i"th target word
                    using var targetIdxTensor = g.Argmax(probs, 1);
                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    for (int j = 0; j < targetIdx.Length; j++)
                    {
                        if (setEndSentId.Contains(j) == false)
                        {
                            outputSnts[j].Add((int)targetIdx[j]);

                            if (targetIdx[j] == eosTokenId)
                            {
                                setEndSentId.Add(j);
                            }
                        }
                    }

                    if (setEndSentId.Count == batchSize)
                    {
                        // All target sentences in current batch are finished, so we exit.
                        break;
                    }

                    ix_inputs = targetIdx;
                }
            }

            return cost;
        }
    }
}
