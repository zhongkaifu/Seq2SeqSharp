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

        public string SrcGroupLenId = "";
        public string TgtGroupLenId = "";


        public int SrcLength = 0;
        public int TgtLength = 0;
        public RawSntPair(string s, string t)
        {
            SrcSnt = s;
            TgtSnt = t;

            SrcLength = CountWhiteSpace(s);
            TgtLength = CountWhiteSpace(t);

            SrcGroupLenId = CountGroupLens(s);
            TgtGroupLenId = CountGroupLens(t);
        }


        private static string CountGroupLens(string s)
        {
            StringBuilder sb = new StringBuilder();
            string[] items = s.Split('\t');

            foreach (var item in items)
            {
                int len = item.Split(' ').Length;

                sb.Append(len.ToString());
                sb.Append("_");
            }

            return sb.ToString();
        }

        private static int CountWhiteSpace(string s)
        {
            string[] items = s.Split(' ');

            return items.Length;


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


        public override void CreateBatch(List<List<List<string>>> srcTokensGroups)
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
