using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class Seq2SeqClassificationCorpusBatch : ISntPairBatch
    {
        public List<List<string>> SrcTkns = null;
        public List<List<List<string>>> TgtTknsGroup = null; // task 0 - generate sequence, task 1 - classify input sequence

        public List<SntPair> SntPairs;

        public virtual string SrcPrefix => BuildInTokens.CLS;
        public virtual string SrcSuffix => "";


        public virtual string TgtPrefix => "";
        public virtual string TgtSuffix => "";

        public int BatchSize => SrcTkns.Count;
        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public Seq2SeqClassificationCorpusBatch()
        {
        }


        public void TryAddPrefix(List<List<string>> tokens, string prefix)
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

        public void TryAddSuffix(List<List<string>> tokens, string suffix)
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
            TgtTknsGroup = new List<List<List<string>>>();

            SrcTokenCount = 0;
            TgtTokenCount = 0;


            for (int i = 0; i < 2; i++)
            {
                TgtTknsGroup.Add(new List<List<string>>());
            }


            for (int i = 0; i < sntPairs.Count; i++)
            {
                SrcTkns.Add(sntPairs[i].SrcSnt.ToList());

                List<string> clsTkns = new List<string>();
                clsTkns.Add(sntPairs[i].TgtSnt[0]);
                TgtTknsGroup[1].Add(clsTkns);


                List<string> tgtTkns = new List<string>();
                tgtTkns.AddRange(sntPairs[i].TgtSnt.ToList().GetRange(1, sntPairs[i].TgtSnt.Length - 1));
                TgtTknsGroup[0].Add(tgtTkns);


                SrcTokenCount += sntPairs[i].SrcSnt.Length;
                TgtTokenCount += (sntPairs[i].TgtSnt.Length - 1);
            }

            TryAddPrefix(SrcTkns, BuildInTokens.CLS);

            TryAddPrefix(TgtTknsGroup[0], BuildInTokens.BOS);
            TryAddSuffix(TgtTknsGroup[0], BuildInTokens.EOS);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {
            SrcTkns = srcTokens;
            TgtTknsGroup = new List<List<List<string>>>();
            TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));


            TryAddPrefix(SrcTkns, BuildInTokens.CLS);
        }

        public ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqClassificationCorpusBatch spb = new Seq2SeqClassificationCorpusBatch();
            spb.SrcTkns = SrcTkns;
            spb.TgtTknsGroup = new List<List<List<string>>>();
            spb.TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));

            return spb;
        }


        public ISntPairBatch GetRange(int idx, int count)
        {
            Seq2SeqClassificationCorpusBatch spb = new Seq2SeqClassificationCorpusBatch();
            spb.SrcTkns = new List<List<string>>();

            spb.SrcTkns.AddRange(SrcTkns.GetRange(idx, count));

            if (TgtTknsGroup != null)
            {
                spb.TgtTknsGroup = new List<List<List<string>>>();
                for (int i = 0; i < TgtTknsGroup.Count; i++)
                {
                    spb.TgtTknsGroup.Add(new List<List<string>>());
                    spb.TgtTknsGroup[i].AddRange(TgtTknsGroup[i].GetRange(idx, count));
                }
            }
            else
            {
                spb.TgtTknsGroup = null;
            }

            return spb;
        }

        public List<List<string>> GetTgtTokens(int group)
        {
            return TgtTknsGroup[group];
        }

        public List<List<string>> GetSrcTokens(int group)
        {
            return SrcTkns;
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
    }
}
