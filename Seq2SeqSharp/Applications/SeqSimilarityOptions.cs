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
    public class SeqSimilarityOptions : Options
    {

        [Arg("It indicates if the embedding is trainable", "IsEmbeddingTrainable")]
        public bool IsEmbeddingTrainable = true;

        [Arg("Maxmium sentence length in valid and test set", "MaxTestSentLength")]
        public int MaxTestSentLength = 32;

        [Arg("Maxmium sentence length in training corpus", "MaxTrainSentLength")]
        public int MaxTrainSentLength = 110;


        [Arg("The type of similarity. Value: Continuous, Discrete. Continuous is by default.", "SimilarityType")]
        public string SimilarityType = "Continuous";

        public DecodingOptions CreateDecodingOptions()
        {
            DecodingOptions decodingOptions = new DecodingOptions();
            decodingOptions.DecodingStrategy = DecodingStrategy;
            decodingOptions.RepeatPenalty = DecodingRepeatPenalty;

            decodingOptions.BeamSearchSize = BeamSearchSize;

            decodingOptions.MaxSrcSentLength = MaxTestSentLength;
            decodingOptions.MaxTgtSentLength = MaxTestSentLength;

            return decodingOptions;
        }
    }
}
