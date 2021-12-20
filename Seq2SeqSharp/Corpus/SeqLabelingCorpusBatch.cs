using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class SeqLabelingCorpusBatch : CorpusBatch
    {
        public override void CreateBatch(List<SntPair> sntPairs)
        {
            base.CreateBatch(sntPairs);
        }


        public override void CreateBatch(List<List<List<string>>> srcTokensGroups, List<List<List<string>>> tgtTokensGroups = null)
        {
            SrcTknsGroups = srcTokensGroups;
            TgtTknsGroups = new List<List<List<string>>>
            {
                InitializeHypTokens("")
            };
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqCorpusBatch spb = new Seq2SeqCorpusBatch
            {
                SrcTknsGroups = SrcTknsGroups,
                TgtTknsGroups = new List<List<List<string>>>()
            };
            spb.TgtTknsGroups.Add(InitializeHypTokens(""));

            return spb;
        }
    }
}
