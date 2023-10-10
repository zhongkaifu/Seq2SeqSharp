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
using System.Linq;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class SeqClassificationModel : Model
    {
        public SeqClassificationModel() { }
        public SeqClassificationModel(SeqClassificationOptions opts, Vocab srcVocab, Vocab tgtVocab)
            : base(opts, srcVocab, tgtVocab)
        {
        }
        public SeqClassificationModel(Model_4_ProtoBufSerializer m)
            : base(m)
        {
        }
        public static SeqClassificationModel Create(Model_4_ProtoBufSerializer m) => new SeqClassificationModel(m);
    }
}
