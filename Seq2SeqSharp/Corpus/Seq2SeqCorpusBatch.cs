using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class RawSntPair
    {
        public string SrcSnt;
        public string TgtSnt;

        public int SrcLength = 0;
        public int TgtLength = 0;
        public RawSntPair(string s, string t)
        {
            SrcSnt = s;
            TgtSnt = t;

            SrcLength = CountWhiteSpace(s);
            TgtLength = CountWhiteSpace(t);

        }

        private int CountWhiteSpace(string s)
        {
            if (String.IsNullOrEmpty(s))
            {
                return 0;
            }

            int cnt = 1;
            bool prevIsSpace = false;
            foreach (char ch in s)
            {
                if (ch == ' ' && prevIsSpace == false)
                {
                    cnt++;
                    prevIsSpace = true;
                }
                else
                {
                    prevIsSpace = false;
                }
            }

            return cnt;

        }

        public bool IsEmptyPair()
        {
            return String.IsNullOrEmpty(SrcSnt) && String.IsNullOrEmpty(TgtSnt);
        }
    }

    public class SntPair
    {
        public string[] SrcSnt;
        public string[] TgtSnt;
    }


    public interface ISntPairBatch
    {
        int BatchSize { get; }
        int SrcTokenCount { get; set; }
        int TgtTokenCount { get; set; }

        string SrcPrefix { get; }
        string SrcSuffix { get; }

        string TgtPrefix { get; }

        string TgtSuffix { get; }

        void CreateBatch(List<SntPair> sntPairs);
        ISntPairBatch CloneSrcTokens();

        ISntPairBatch GetRange(int idx, int count);

        List<List<string>> GetSrcTokens(int group);
        List<List<string>> GetTgtTokens(int group);

    }

    public class Seq2SeqCorpusBatch : ISntPairBatch
    {
        public List<List<string>> SrcTkns = null;
        public List<List<string>> TgtTkns = null;

        public List<SntPair> SntPairs;

        public virtual string SrcPrefix => BuildInTokens.BOS;
        public virtual string SrcSuffix => BuildInTokens.EOS;


        public virtual string TgtPrefix => BuildInTokens.BOS;
        public virtual string TgtSuffix => BuildInTokens.EOS;

        public int BatchSize => SrcTkns.Count;
        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public Seq2SeqCorpusBatch()
        {
        }


        public void AddPrefix(List<List<string>> tokens, string prefix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(prefix);
                }
                else
                {
                    if (tokens[i][0] != prefix)
                    {
                        tokens[i].Insert(0, prefix);
                    }
                }
            }
        }


        public void AddSuffix(List<List<string>> tokens, string suffix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(suffix);
                }
                else
                {
                    if (tokens[i][tokens[i].Count - 1] != suffix)
                    {
                        tokens[i].Add(suffix);
                    }
                }
            }
        }


        public void CreateBatch(List<SntPair> sntPairs)
        {
            SntPairs = sntPairs;

            SrcTkns = new List<List<string>>();
            TgtTkns = new List<List<string>>();
            SrcTokenCount = 0;
            TgtTokenCount = 0;

            for (int i = 0; i < sntPairs.Count; i++)
            {
                SrcTkns.Add(sntPairs[i].SrcSnt.ToList());
                TgtTkns.Add(sntPairs[i].TgtSnt.ToList());

                SrcTokenCount += sntPairs[i].SrcSnt.Length;
                TgtTokenCount += sntPairs[i].TgtSnt.Length;
            }

            AddPrefix(SrcTkns, SrcPrefix);
            AddSuffix(SrcTkns, SrcSuffix);
            AddPrefix(TgtTkns, TgtPrefix);
            AddSuffix(TgtTkns, TgtSuffix);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {
            SrcTkns = srcTokens;
            TgtTkns = InitializeHypTokens(TgtPrefix);


            AddPrefix(SrcTkns, SrcPrefix);
            AddSuffix(SrcTkns, SrcSuffix);
        }

        public ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch();
            spb.SrcTkns = SrcTkns;
            spb.TgtTkns = InitializeHypTokens(TgtPrefix);

            return spb;
        }


        public ISntPairBatch GetRange(int idx, int count)
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch();
            spb.SrcTkns = new List<List<string>>();
            spb.TgtTkns = new List<List<string>>();

            spb.SrcTkns.AddRange(SrcTkns.GetRange(idx, count));
            spb.TgtTkns.AddRange(TgtTkns.GetRange(idx, count));

            return spb;
        }

        List<List<string>> InitializeHypTokens(string prefix)
        {
            List<List<string>> hypTkns = new List<List<string>>();
            for (int i = 0; i < BatchSize; i++)
            {
                if (String.IsNullOrEmpty(prefix) == false)
                {
                    hypTkns.Add(new List<string>() { prefix });
                }
                else
                {
                    hypTkns.Add(new List<string>());
                }
            }

            return hypTkns;
        }

        public List<List<string>> GetTgtTokens(int group)
        {
            return TgtTkns;
        }

        public List<List<string>> GetSrcTokens(int group)
        {
            return SrcTkns;
        }
    }
}
