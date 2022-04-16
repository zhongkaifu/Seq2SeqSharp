// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Applications
{
    public class Encoder
    {
        static List<List<string>> InsertCLSToken(List<List<string>> tokens)
        {
            List<List<string>> newTokens = new List<List<string>>();

            foreach (var item in tokens)
            {
                List<string> r = new List<string>
                {
                    BuildInTokens.CLS
                };
                r.AddRange(item);

                newTokens.Add(r);

            }

            return newTokens;
        }

        public static (MultiProcessorNetworkWrapper<IEncoder>, int) CreateEncoders(IModel modelMetaData, Options options, RoundArray<int> raDeviceIds)
        {
            int contextDim;
            MultiProcessorNetworkWrapper<IEncoder> encoder = null;
            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: options.IsEncoderTrainable), raDeviceIds.ToArray());

                contextDim = modelMetaData.HiddenDim * 2;
            }
            else
            {
                encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EncoderEmbeddingDim, modelMetaData.EncoderLayerDepth, options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: options.IsEncoderTrainable, learningRateFactor: options.EncoderStartLearningRateFactor), raDeviceIds.ToArray());

                contextDim = modelMetaData.HiddenDim;
            }

            return (encoder, contextDim);
        }

        static public IWeightTensor Run(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, IEncoder encoder, IModel modelMetaData, ShuffleEnums shuffleType,
            IWeightTensor srcEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding, List<List<int>> srcSntsIds, float[] originalSrcLengths)
        {
            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSntsIds.Count);

            IWeightTensor encOutput = InnerRunner(computeGraph, srcSntsIds, originalSrcLengths, shuffleType, encoder, modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding);
            return encOutput;
        }

        public static IWeightTensor BuildTensorForSourceTokenGroupAt(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, ShuffleEnums shuffleType, IEncoder encoder, IModel modelMetaData, IWeightTensor srcEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding, int groupId)
        {
            var contextTokens = InsertCLSToken(sntPairBatch.GetSrcTokens(groupId));
            var originalSrcContextLength = BuildInTokens.PadSentences(contextTokens);
            var contextTokenIds = modelMetaData.SrcVocab.GetWordIndex(contextTokens);

            IWeightTensor encContextOutput = InnerRunner(computeGraph, contextTokenIds, originalSrcContextLength, shuffleType, encoder, modelMetaData, srcEmbedding, posEmbedding, segmentEmbedding);

            int contextPaddedLen = contextTokens[0].Count;
            float[] contextCLSIdxs = new float[sntPairBatch.BatchSize];
            for (int j = 0; j < sntPairBatch.BatchSize; j++)
            {
                contextCLSIdxs[j] = j * contextPaddedLen;
            }

            IWeightTensor contextCLSOutput = computeGraph.IndexSelect(encContextOutput, contextCLSIdxs);
            return contextCLSOutput;
        }

        static private IWeightTensor InnerRunner(IComputeGraph computeGraph, List<List<int>> srcTokensList, float[] originalSrcLengths, ShuffleEnums shuffleType, IEncoder encoder, IModel modelMetaData,
           IWeightTensor srcEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding)
        {
            int batchSize = srcTokensList.Count;
            int srcSeqPaddedLen = srcTokensList[0].Count;
            IWeightTensor srcSelfMask = (shuffleType == ShuffleEnums.NoPaddingInSrc || shuffleType == ShuffleEnums.NoPadding || batchSize == 1) ? null : computeGraph.BuildPadSelfMask(srcSeqPaddedLen, originalSrcLengths); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            var encOutput = RunEncoder(computeGraph, srcTokensList, encoder, modelMetaData, srcEmbedding, srcSelfMask, posEmbedding, segmentEmbedding);
            if (srcSelfMask != null)
            {
                srcSelfMask.Dispose();
            }

            return encOutput;
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
        static private IWeightTensor RunEncoder(IComputeGraph g, List<List<int>> seqs, IEncoder encoder, IModel modelMetaData, IWeightTensor embeddings, IWeightTensor selfMask, IWeightTensor posEmbeddings, 
            IWeightTensor segmentEmbeddings)
        {
            int batchSize = seqs.Count;
            var inputEmbs = TensorUtils.CreateTokensEmbeddings(seqs, g, embeddings, segmentEmbeddings, modelMetaData.SrcVocab, (float)Math.Sqrt(embeddings.Columns), enableTagEmbedding: modelMetaData.EnableTagEmbeddings);

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, 0.0f);
            }

            return encoder.Encode(inputEmbs, batchSize, g, selfMask);
        }
    }
}
