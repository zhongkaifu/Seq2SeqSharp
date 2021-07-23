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
        public List<List<string>> SrcTokenGroups;
        public List<List<string>> TgtTokenGroups;

        public SntPair(string srcLine, string tgtLine)
        {
            SrcTokenGroups = new List<List<string>>();
            TgtTokenGroups = new List<List<string>>();

            CreateGroup(srcLine, SrcTokenGroups);
            CreateGroup(tgtLine, TgtTokenGroups);
        }

        private void CreateGroup(string line, List<List<string>> sntGroup)
        {
            string[] groups = line.Split('\t');
            foreach (var group in groups)
            {
                sntGroup.Add(group.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList());
            }
        }
    }


    public interface ISntPairBatch
    {
        int BatchSize { get; }
        int SrcTokenCount { get; set; }
        int TgtTokenCount { get; set; }

        void CreateBatch(List<SntPair> sntPairs);
        ISntPairBatch CloneSrcTokens();

        ISntPairBatch GetRange(int idx, int count);

        List<List<string>> GetSrcTokens(int group);
        List<List<string>> GetTgtTokens(int group);

    }

    public class Seq2SeqCorpusBatch : CorpusBatch
    {

        public override void CreateBatch(List<SntPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcTknsGroup[0], BuildInTokens.BOS);
            TryAddSuffix(SrcTknsGroup[0], BuildInTokens.EOS);
            TryAddPrefix(TgtTknsGroup[0], BuildInTokens.BOS);
            TryAddSuffix(TgtTknsGroup[0], BuildInTokens.EOS);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {

            SrcTknsGroup = new List<List<List<string>>>();
            SrcTknsGroup.Add(srcTokens);

            TgtTknsGroup = new List<List<List<string>>>();
            TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch();
            spb.SrcTknsGroup = SrcTknsGroup;
            spb.TgtTknsGroup = new List<List<List<string>>>();
            spb.TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));

            return spb;
        }
    }
}
