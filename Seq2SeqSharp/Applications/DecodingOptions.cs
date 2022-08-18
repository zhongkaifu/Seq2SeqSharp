// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Applications
{
    public class DecodingOptions
    {
        public int MaxSrcSentLength = 128;

        public int MaxTgtSentLength = 256;

        // The penalty for decoded repeat tokens. Default is 5.0
        public float RepeatPenalty = 5.0f;

        // Beam search size. Default is 1
        public int BeamSearchSize = 1;

        // Decoding strategies. It supports GreedySearch and Sampling. Default is GreedySearch
        public DecodingStrategyEnums DecodingStrategy = DecodingStrategyEnums.GreedySearch;

        // It indicates if aligments to source sequence should be outputted
        public bool OutputAligmentsToSrc = false;

    }
}
