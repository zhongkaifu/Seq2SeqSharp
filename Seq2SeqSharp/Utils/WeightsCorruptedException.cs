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

namespace Seq2SeqSharp.Utils
{
    public class WeightsCorruptedException : Exception
    {
        public WeightsCorruptedException() { } 

        public WeightsCorruptedException(string message) : base(message) { }       
    }

    public class GradientsCorruptedException : Exception
    {
        public GradientsCorruptedException() { }

        public GradientsCorruptedException(string message) : base(message) { }
    }
}
