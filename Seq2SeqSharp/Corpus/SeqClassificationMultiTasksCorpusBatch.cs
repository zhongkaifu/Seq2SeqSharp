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

        public override void CreateBatch(List<IPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcBatchTokens, BuildInTokens.CLS);
        }


        public override void CreateBatch(List<List<string>> srcTokens, List<List<string>> tgtTokens = null)
        {
            SrcBatchTokens = srcTokens;
            TgtBatchTokens = null;


            TryAddPrefix(SrcBatchTokens, BuildInTokens.CLS);
        }

        public override ISntPairBatch CloneSrcTokens()
        {
            SeqClassificationMultiTasksCorpusBatch spb = new SeqClassificationMultiTasksCorpusBatch
            {
                SrcBatchTokens = SrcBatchTokens,
                TgtBatchTokens = null
            };

            return spb;
        }
    }
}
