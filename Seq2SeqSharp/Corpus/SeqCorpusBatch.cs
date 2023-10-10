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
    public class SeqCorpusBatch : CorpusBatch
    {

        public override void CreateBatch(List<IPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcBatchTokens, BuildInTokens.BOS);
            TryAddSuffix(SrcBatchTokens, BuildInTokens.EOS);
            TryAddPrefix(TgtBatchTokens, BuildInTokens.BOS);
            TryAddSuffix(TgtBatchTokens, BuildInTokens.EOS);
        }


        public override void CreateBatch(List<List<string>> srcTokens, List<List<string>> tgtTokens)
        {

            SrcBatchTokens = srcTokens;

            TryAddPrefix(SrcBatchTokens, BuildInTokens.BOS);
            //TryAddSuffix(SrcTknsGroups[0], BuildInTokens.EOS);


            if (tgtTokens != null)
            {
                TgtBatchTokens = tgtTokens;
                TryAddPrefix(TgtBatchTokens, BuildInTokens.BOS);
            }
            else
            {
                TgtBatchTokens = InitializeHypTokens(BuildInTokens.BOS);                
            }
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch
            {
                SrcBatchTokens = SrcBatchTokens,
                TgtBatchTokens = InitializeHypTokens(BuildInTokens.BOS)
            };

            return spb;
        }
    }
}
