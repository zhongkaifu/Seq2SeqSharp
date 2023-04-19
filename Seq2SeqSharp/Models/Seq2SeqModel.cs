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
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModel : Model
    {
        public Seq2SeqModel() { }

        public Seq2SeqModel(Seq2SeqOptions opts, Vocab srcVocab, Vocab tgtVocab)
            : base(opts, srcVocab)
        {
            DecoderEmbeddingDim = opts.TgtEmbeddingDim;
            DecoderLayerDepth = opts.DecoderLayerDepth;
            DecoderType = opts.DecoderType;
            EnableCoverageModel = opts.EnableCoverageModel;
            SharedEmbeddings = opts.SharedEmbeddings;
            TgtVocab = tgtVocab;
            PointerGenerator = opts.PointerGenerator;
        }

        public Seq2SeqModel(Model_4_ProtoBufSerializer m)
            : base(m)
        {
            ClsVocabs = m.ClsVocabs?.Select(v => v.ToVocab()).ToList();

            DecoderEmbeddingDim = m.DecoderEmbeddingDim;
            DecoderLayerDepth = m.DecoderLayerDepth;
            DecoderType = m.DecoderType;
            EnableCoverageModel = m.EnableCoverageModel;
            SharedEmbeddings = m.SharedEmbeddings;
            TgtVocab = m.TgtVocab?.ToVocab();
        }
        public static Seq2SeqModel Create(Model_4_ProtoBufSerializer m) => new Seq2SeqModel(m);
    }
}
