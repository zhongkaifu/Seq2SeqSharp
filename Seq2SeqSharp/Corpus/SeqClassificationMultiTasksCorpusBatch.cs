using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class SeqClassificationMultiTasksCorpusBatch : ISntPairBatch
    {
        public List<List<string>> SrcTkns = null;
        public List<List<List<string>>> TgtTknsGroup = null;

        public List<SntPair> SntPairs;

        public virtual string SrcPrefix => BuildInTokens.CLS;
        public virtual string SrcSuffix => "";


        public virtual string TgtPrefix => "";
        public virtual string TgtSuffix => "";

        public int BatchSize => SrcTkns.Count;
        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public SeqClassificationMultiTasksCorpusBatch()
        {
        }


        private void AddPrefix(List<List<string>> tokens, string prefix)
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

        public void CreateBatch(List<SntPair> sntPairs)
        {
            SntPairs = sntPairs;

            SrcTkns = new List<List<string>>();
            TgtTknsGroup = new List<List<List<string>>>();

            SrcTokenCount = 0;
            TgtTokenCount = 0;


            for (int i = 0; i < sntPairs[0].TgtSnt.Length; i++)
            {
                TgtTknsGroup.Add(new List<List<string>>());
            }


            for (int i = 0; i < sntPairs.Count; i++)
            {
                SrcTkns.Add(sntPairs[i].SrcSnt.ToList());


                for (int j = 0; j < sntPairs[i].TgtSnt.Length; j++)
                {
                    List<string> tgtTkn = new List<string>();
                    tgtTkn.Add(sntPairs[i].TgtSnt[j]);
                    TgtTknsGroup[j].Add(tgtTkn);
                }

                SrcTokenCount += sntPairs[i].SrcSnt.Length;
                TgtTokenCount++;
            }

            AddPrefix(SrcTkns, SrcPrefix);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {
            SrcTkns = srcTokens;
            TgtTknsGroup = null;


            AddPrefix(SrcTkns, SrcPrefix);
        }

        public ISntPairBatch CloneSrcTokens()
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch();
            spb.SrcTkns = SrcTkns;
            spb.TgtTknsGroup = null;

            return spb;
        }


        public ISntPairBatch GetRange(int idx, int count)
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch();
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
    }
}
