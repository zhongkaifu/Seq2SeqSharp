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

namespace Seq2SeqSharp.Applications
{
    public class SeqClassificationOptions : Options
    {

        [Arg("The embedding dim", "EmbeddingDim")]
        public int EmbeddingDim = 128;

        [Arg("It indicates if the embedding is trainable", "IsEmbeddingTrainable")]
        public bool IsEmbeddingTrainable = true;

        [Arg("Maxmium sentence length", nameof(MaxSentLength))]
        public int MaxSentLength = 110;

        public DecodingOptions CreateDecodingOptions()
        {
            DecodingOptions decodingOptions = new DecodingOptions();
            decodingOptions.DecodingStrategy = DecodingStrategy;
            decodingOptions.TopPValue = DecodingTopPValue;
            decodingOptions.RepeatPenalty = DecodingRepeatPenalty;

            decodingOptions.BeamSearchSize = BeamSearchSize;

            decodingOptions.MaxSrcSentLength = MaxSentLength;
            decodingOptions.MaxTgtSentLength = MaxSentLength;

            return decodingOptions;
        }
    }
}
