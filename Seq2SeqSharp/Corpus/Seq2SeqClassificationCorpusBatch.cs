using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class Seq2SeqClassificationCorpusBatch : CorpusBatch
    {

        public override void CreateBatch(List<SntPair> sntPairs)
        {
            base.CreateBatch(sntPairs);

            TryAddPrefix(SrcTknsGroup[0], BuildInTokens.CLS);

            TryAddPrefix(TgtTknsGroup[1], BuildInTokens.BOS);
            TryAddSuffix(TgtTknsGroup[1], BuildInTokens.EOS);
        }


        public void CreateBatch(List<List<string>> srcTokens)
        {
            SrcTknsGroup = new List<List<List<string>>>();
            SrcTknsGroup.Add(srcTokens);

            TgtTknsGroup = new List<List<List<string>>>();
            TgtTknsGroup.Add(new List<List<string>>());
            TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));


            TryAddPrefix(SrcTknsGroup[0], BuildInTokens.CLS);
        }


        public override ISntPairBatch CloneSrcTokens()
        {
            Seq2SeqClassificationCorpusBatch spb = new Seq2SeqClassificationCorpusBatch();
            spb.SrcTknsGroup = SrcTknsGroup;
            spb.TgtTknsGroup = new List<List<List<string>>>();
            spb.TgtTknsGroup.Add(new List<List<string>>());
            spb.TgtTknsGroup.Add(InitializeHypTokens(BuildInTokens.BOS));

            return spb;
        }



    }
}
