using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public interface IPairBatch
    {
        int BatchSize { get; }
        int SrcTokenCount { get; set; }
        int TgtTokenCount { get; set; }

        IPairBatch GetRange(int idx, int count);
        IPairBatch CloneSrcTokens();

        List<List<string>> GetSrcTokens();
        List<List<string>> GetTgtTokens();

        void CreateBatch(List<IPair> sntPairs);

        void CreateBatch(List<List<string>> srcTokens, List<List<string>> tgtTokens);

        int GetTgtGroupSize();
    }

    public interface ISntPairBatch : IPairBatch
    {


        int GetSrcGroupSize();


    }


    public interface IVisionSntPairBatch : IPairBatch
    {
    }
}
