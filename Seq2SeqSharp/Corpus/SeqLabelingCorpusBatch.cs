// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;

namespace Seq2SeqSharp.Corpus
{
    public class SeqLabelingCorpusBatch : CorpusBatch
    {
        public override void CreateBatch(List<IPair> sntPairs)
        {
            base.CreateBatch(sntPairs);
        }


        public override void CreateBatch(List<List<string>> srcTokens, List<List<string>> tgtTokens = null)
        {
            SrcBatchTokens = srcTokens;
            TgtBatchTokens = InitializeHypTokens("");
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch
            {
                SrcBatchTokens = SrcBatchTokens,
                TgtBatchTokens = InitializeHypTokens("")
            };

            return spb;
        }
    }
}
