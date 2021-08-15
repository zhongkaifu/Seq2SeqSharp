using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public interface ISntPairBatch
    {
        int BatchSize { get; }
        int SrcTokenCount { get; set; }
        int TgtTokenCount { get; set; }

        void CreateBatch(List<SntPair> sntPairs);
        void CreateBatch(List<List<List<string>>> srcTokensGroups);


        ISntPairBatch CloneSrcTokens();

        ISntPairBatch GetRange(int idx, int count);

        List<List<string>> GetSrcTokens(int group);
        List<List<string>> GetTgtTokens(int group);

        int GetSrcGroupSize();
        int GetTgtGroupSize();

    }
}
