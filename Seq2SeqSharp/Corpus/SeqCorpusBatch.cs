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

        public override void CreateBatch(List<SntPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcTknsGroups[0], BuildInTokens.BOS);
            TryAddSuffix(SrcTknsGroups[0], BuildInTokens.EOS);
            TryAddPrefix(TgtTknsGroups[0], BuildInTokens.BOS);
            TryAddSuffix(TgtTknsGroups[0], BuildInTokens.EOS);
        }


        public override void CreateBatch(List<List<List<string>>> srcTokensGroups, List<List<List<string>>> tgtTokensGroups)
        {

            SrcTknsGroups = srcTokensGroups;

            TryAddPrefix(SrcTknsGroups[0], BuildInTokens.BOS);
            //TryAddSuffix(SrcTknsGroups[0], BuildInTokens.EOS);


            if (tgtTokensGroups != null)
            {
                TgtTknsGroups = tgtTokensGroups;
                TryAddPrefix(TgtTknsGroups[0], BuildInTokens.BOS);
            }
            else
            {
                TgtTknsGroups = new List<List<List<string>>>
               {
                    InitializeHypTokens(BuildInTokens.BOS)
                };
            }
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch
            {
                SrcTknsGroups = SrcTknsGroups,
                TgtTknsGroups = new List<List<List<string>>>()
            };
            spb.TgtTknsGroups.Add(InitializeHypTokens(BuildInTokens.BOS));

            return spb;
        }
    }
}
