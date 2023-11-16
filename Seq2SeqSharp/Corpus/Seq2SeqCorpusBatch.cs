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
            if (s == null)
            {
                s = "";
            }

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


    public interface IPair
    {
        int GetSrcTokenCount();
        int GetTgtTokenCount();

    }

    public class SntPair : IPair
    {
        public List<string> SrcTokens;
        public List<string> TgtTokens;

        public SntPair(string srcLine, string tgtLine)
        {
            SrcTokens = new List<string>();
            TgtTokens = new List<string>();

            if (String.IsNullOrEmpty(srcLine) == false)
            {
                SrcTokens = srcLine.Split(' ').ToList();
            }

            if (String.IsNullOrEmpty(tgtLine) == false)
            {
                TgtTokens = tgtLine.Split(' ').ToList();
            }
        }

        public int GetTgtTokenCount()
        {
            return TgtTokens.Count;
        }

        public int GetSrcTokenCount()
        {
            return SrcTokens.Count;
        }

        public string PrintSrcTokens()
        {
            return string.Join(" ", SrcTokens);
        }

        public string PrintTgtTokens()
        {
            return string.Join(" ", TgtTokens);
        }

    }




    public class Seq2SeqCorpusBatch : CorpusBatch
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
            TryAddSuffix(SrcBatchTokens, BuildInTokens.EOS);


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
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch();
            spb.SrcBatchTokens = SrcBatchTokens;
            spb.TgtBatchTokens = InitializeHypTokens(BuildInTokens.BOS);


            return spb;
        }
    }
}
