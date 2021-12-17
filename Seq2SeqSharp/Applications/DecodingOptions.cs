using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class DecodingOptions
    {
        public int MaxSrcSentLength = 128;

        public int MaxTgtSentLength = 256;

        // The top-P value for sampling decoding strategy. The value above 0.0 will cause non-deterministic results. Default is 0.0
        public float TopPValue = 0.0f;

        // The penalty for decoded repeat tokens. Default is 2.0
        public float RepeatPenalty = 2.0f;

        // "The penalty for decoded token distance. Default is 5.0
        public float DistancePenalty = 5.0f;

        // Beam search size. Default is 1
        public int BeamSearchSize = 1;

        // Decoding strategies. It supports GreedySearch and Sampling. Default is GreedySearch
        public DecodingStrategyEnums DecodingStrategy = DecodingStrategyEnums.GreedySearch;

    }
}
