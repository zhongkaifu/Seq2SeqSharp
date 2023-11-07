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
using System.ComponentModel.DataAnnotations;

namespace Seq2SeqSharp.Applications
{
    public class SeqClassificationOptions : Options
    {

        [Arg("It indicates if the embedding is trainable", nameof(IsEmbeddingTrainable))]
        public bool IsEmbeddingTrainable = true;

        [Arg("Maxmium sentence length", nameof(MaxSentLength))]
        [Range(1, 99999)]
        public int MaxSentLength = 110;


        [Arg("The weights of each tag while calculating loss during training. Format: Tag1:Weight1, Tag2:Weight2, ... ,TagN:WeightN", nameof(TagWeights))]
        public string TagWeights = "";

        public DecodingOptions CreateDecodingOptions()
        {
            DecodingOptions decodingOptions = new DecodingOptions();
            decodingOptions.DecodingStrategy = DecodingStrategy;
            decodingOptions.TopP = DecodingTopP;
            decodingOptions.RepeatPenalty = DecodingRepeatPenalty;
            decodingOptions.Temperature = DecodingTemperature;
            decodingOptions.BeamSearchSize = BeamSearchSize;

            decodingOptions.MaxSrcSentLength = MaxSentLength;
            decodingOptions.MaxTgtSentLength = MaxSentLength;

            return decodingOptions;
        }
    }
}
