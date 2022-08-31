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
using System.Linq;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class Seq2SeqClassificationModel : Seq2SeqModel
    {
        public Seq2SeqClassificationModel() { }
        public Seq2SeqClassificationModel(Seq2SeqClassificationOptions opts, Vocab srcVocab, Vocab tgtVocab, Vocab clsVocab)
            : base(opts, srcVocab, tgtVocab)
        {
            ClsVocab = clsVocab;
        }
        public Seq2SeqClassificationModel(Model_4_ProtoBufSerializer m)
            : base(m)
        {
            ClsVocabs = m.ClsVocabs?.Select(v => v.ToVocab()).ToList();
            Name2Weights = m.Name2Weights;
        }
        public static new Seq2SeqClassificationModel Create(Model_4_ProtoBufSerializer m) => new Seq2SeqClassificationModel(m);
    }
}
