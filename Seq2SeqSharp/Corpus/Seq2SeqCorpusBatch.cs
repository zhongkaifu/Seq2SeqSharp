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

        private static int CountWhiteSpace(string s)
        {
            string[] items = s.Split(' ');

            return items.Length;

            //if (String.IsNullOrEmpty(s))
            //{
            //    return 0;
            //}

            //int cnt = 1;
            //bool prevIsSpace = false;
            //foreach (char ch in s)
            //{
            //    if (ch == ' ' && prevIsSpace == false)
            //    {
            //        cnt++;
            //        prevIsSpace = true;
            //    }
            //    else
            //    {
            //        prevIsSpace = false;
            //    }
            //}

            //return cnt;
        }

        public bool IsEmptyPair()
        {
            return String.IsNullOrEmpty(SrcSnt) && String.IsNullOrEmpty(TgtSnt);
        }
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
                rst.Add($"GroupId '{gIdx}': " + String.Join(" ", g));
                gIdx++;
            }

            return String.Join("\n", rst);
        }

        public string PrintTgtTokens()
        {
            List<string> rst = new List<string>();
            int gIdx = 0;
            foreach (var g in TgtTokenGroups)
            {
                rst.Add($"GroupId '{gIdx}': " + String.Join(" ", g));
                gIdx++;
            }

            return String.Join("\n", rst);
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

        int GetSrcGroupSize();
        int GetTgtGroupSize();

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


        public void CreateBatch(List<List<List<string>>> srcTokensGroups)
        {

            SrcTknsGroups = srcTokensGroups;

            TryAddPrefix(SrcTknsGroups[0], BuildInTokens.BOS);
            TryAddSuffix(SrcTknsGroups[0], BuildInTokens.EOS);


            TgtTknsGroups = new List<List<List<string>>>
            {
                InitializeHypTokens(BuildInTokens.BOS)
            };
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
