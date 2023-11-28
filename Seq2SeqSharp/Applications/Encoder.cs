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
using TensorSharp;
using Seq2SeqSharp.Enums;

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

        public static MultiProcessorNetworkWrapper<IEncoder> CreateEncoders(IModel model, Options options, RoundArray<int> raDeviceIds, DType elementType = DType.Float32)
        {
            MultiProcessorNetworkWrapper<IEncoder> encoder = null;
            if (model.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", model.HiddenDim, model.EncoderEmbeddingDim, model.EncoderLayerDepth, raDeviceIds.GetNextItem(), isTrainable: options.IsEncoderTrainable), raDeviceIds.ToArray());
            }
            else
            {
                encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", model.MultiHeadNum, model.HiddenDim, model.IntermediateDim, model.EncoderEmbeddingDim, model.EncoderLayerDepth, options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: options.IsEncoderTrainable, learningRateFactor: options.EncoderStartLearningRateFactor, activateFunc: model.ActivateFunc, expertNum: model.ExpertNum, expertsPerTokenFactor: model.ExpertsPerTokenFactor, 
                    elementType, peType: model.PEType, normType: model.NormType), raDeviceIds.ToArray());
            }

            return encoder;
        }

        static public IWeightTensor Run(IComputeGraph computeGraph, IEncoder encoder, IModel modelMetaData, PaddingEnums paddingType,
            IWeightTensor srcEmbedding, IWeightTensor posEmbeddings, IWeightTensor segmentEmbedding, List<List<int>> srcSntsIds, float[] originalSrcLengths, bool amp = false)
        {
            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSntsIds.Count);

            IWeightTensor encOutput = InnerRunner(computeGraph, srcSntsIds, originalSrcLengths, paddingType, encoder, modelMetaData, srcEmbedding, posEmbeddings, segmentEmbedding, amp);
            return encOutput;
        }

        static private IWeightTensor InnerRunner(IComputeGraph computeGraph, List<List<int>> srcTokensList, float[] originalSrcLengths, PaddingEnums paddingType, IEncoder encoder, IModel modelMetaData,
           IWeightTensor srcEmbedding, IWeightTensor posEmbedding, IWeightTensor segmentEmbedding, bool amp = false)
        {
            int batchSize = srcTokensList.Count;
            int srcSeqPaddedLen = srcTokensList[0].Count;
            IWeightTensor srcSelfMask = (paddingType == PaddingEnums.NoPaddingInSrc || paddingType == PaddingEnums.NoPadding || batchSize == 1) ? null : computeGraph.BuildPadSelfMask(srcSeqPaddedLen, originalSrcLengths, elementType: amp ? DType.Float16 : DType.Float32); // The length of source sentences are same in a single mini-batch, so we don't have source mask.

            // Encoding input source sentences
            var encOutput = RunEncoder(computeGraph, srcTokensList, encoder, modelMetaData, srcEmbedding, srcSelfMask, posEmbedding, segmentEmbedding, amp: amp);
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
            IWeightTensor segmentEmbeddings, bool amp = false)
        {
            int batchSize = seqs.Count;
            var inputEmbs = TensorUtils.CreateTokensEmbeddings(seqs, g, embeddings, segmentEmbeddings, modelMetaData.SrcVocab, (float)Math.Sqrt(embeddings.Columns), enableTagEmbedding: modelMetaData.EnableTagEmbeddings, amp: amp);

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer && posEmbeddings != null)
            {
                inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, 0.0f);
            }

            return encoder.Encode(inputEmbs, batchSize, g, selfMask);
        }
    }
}
