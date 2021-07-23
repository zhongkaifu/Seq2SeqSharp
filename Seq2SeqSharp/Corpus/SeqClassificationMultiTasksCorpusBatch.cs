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

            TryAddPrefix(SrcTknsGroup[0], BuildInTokens.CLS);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {
            SrcTknsGroup = new List<List<List<string>>>();
            SrcTknsGroup.Add(srcTokens);
            TgtTknsGroup = null;


            TryAddPrefix(SrcTknsGroup[0], BuildInTokens.CLS);
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch();
            spb.SrcTknsGroup = SrcTknsGroup;
            spb.TgtTknsGroup = null;

            return spb;
        }
    }
}
