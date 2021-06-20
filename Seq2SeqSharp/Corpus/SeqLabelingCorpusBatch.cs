using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class SeqLabelingCorpusBatch : Seq2SeqCorpusBatch
    {
        public override string SrcPrefix => "";
        public override string SrcSuffix => "";


        public override string TgtPrefix => "";
        public override string TgtSuffix => "";

    }
}
