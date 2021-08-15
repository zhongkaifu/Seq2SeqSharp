using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class SeqClassificationMultiTasksCorpusBatch : CorpusBatch
    {

        public override void CreateBatch(List<SntPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcTknsGroups[0], BuildInTokens.CLS);
        }


        public override void CreateBatch(List<List<List<string>>> srcTokensGroups)
        {
            SrcTknsGroups = srcTokensGroups;
            TgtTknsGroups = null;


            TryAddPrefix(SrcTknsGroups[0], BuildInTokens.CLS);
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch
            {
                SrcTknsGroups = SrcTknsGroups,
                TgtTknsGroups = null
            };

            return spb;
        }
    }
}
