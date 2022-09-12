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
using System;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp.Corpus
{
    public class RawSntPair
    {
        public string SrcSnt;
        public string TgtSnt;

        public long SrcGroupLenId = 0;
        public long TgtGroupLenId = 0;
        public long GroupLenId = 0;

        public int SrcTokenSize = 0;
        public int TgtTokenSize = 0;

        private long maxSeqLength = 0;
        public RawSntPair(string s, string t, int maxSrcSeqLength, int maxTgtSeqLength, bool truncateTooLongSeq)
        {
            this.maxSeqLength = Math.Max(maxSrcSeqLength, maxTgtSeqLength);

            if (truncateTooLongSeq)
            {
                s = TruncateSeq(s, maxSrcSeqLength);
                t = TruncateSeq(t, maxTgtSeqLength);
            }

            SrcTokenSize = CountWhiteSpace(s);
            TgtTokenSize = CountWhiteSpace(t);

            SrcGroupLenId = GenerateGroupLenId(s);
            TgtGroupLenId = GenerateGroupLenId(t);
            GroupLenId = GenerateGroupLenId(s + "\t" + t);

            SrcSnt = s;
            TgtSnt = t;
        }

        public string TruncateSeq(string str, int maxSeqLength)
        {
            string[] items = str.Split('\t');
            List<string> results = new List<string>();

            foreach (var item in items)
            {
                string[] tokens = item.Split(' ');

                if (tokens.Length <= maxSeqLength)
                {
                    results.Add(item);
                }
                else
                {
                    results.Add(string.Join(' ', tokens, 0, maxSeqLength));
                }
            }

            return string.Join("\t", results);
        }

        private long GenerateGroupLenId(string s)
        {
            long r = 0;
            string[] items = s.Split('\t');

            foreach (var item in items)
            {
                r = r * maxSeqLength;

                int len = item.Split(' ').Length;
                r += len;
            }

            return r;
        }

        private int CountWhiteSpace(string s)
        {
            string[] items = s.Split(' ');

            return items.Length;


        }
        public bool IsEmptyPair() => SrcSnt.IsNullOrEmpty() && TgtSnt.IsNullOrEmpty();
    }

    public class SntPair
    {
        public List<List<string>> SrcTokenGroups; //shape: (group_size, sequence_length)
        public List<List<string>> TgtTokenGroups; //shape: (group_size, sequence_length)

        public SntPair(string srcLine, string tgtLine)
        {
            SrcTokenGroups = new List<List<string>>();
            TgtTokenGroups = new List<List<string>>();

            CreateGroup(srcLine, SrcTokenGroups);
            CreateGroup(tgtLine, TgtTokenGroups);
        }

        private static void CreateGroup(string line, List<List<string>> sntGroup)
        {
            string[] groups = line.Split('\t');
            foreach (var group in groups)
            {
                sntGroup.Add(group.Split(' ').ToList());
            }
        }

        public string PrintSrcTokens()
        {
            List<string> rst = new List<string>();
            int gIdx = 0;
            foreach (var g in SrcTokenGroups)
            {
                rst.Add($"GroupId '{gIdx}': " + string.Join(" ", g));
                gIdx++;
            }

            return string.Join("\n", rst);
        }

        public string PrintTgtTokens()
        {
            List<string> rst = new List<string>();
            int gIdx = 0;
            foreach (var g in TgtTokenGroups)
            {
                rst.Add($"GroupId '{gIdx}': " + string.Join(" ", g));
                gIdx++;
            }

            return string.Join("\n", rst);
        }

    }




    public class Seq2SeqCorpusBatch : CorpusBatch
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
            TryAddSuffix(SrcTknsGroups[0], BuildInTokens.EOS);


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
