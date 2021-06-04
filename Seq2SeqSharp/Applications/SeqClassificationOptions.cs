using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class SeqClassificationOptions : Options
    {

        [Arg("The embedding dim", "EmbeddingDim")]
        public int EmbeddingDim = 128;

        [Arg("It indicates if the embedding is trainable", "IsEmbeddingTrainable")]
        public bool IsEmbeddingTrainable = true;

        [Arg("Maxmium sentence length in valid and test set", "MaxTestSentLength")]
        public int MaxTestSentLength = 32;

        [Arg("Maxmium sentence length in training corpus", "MaxTrainSentLength")]
        public int MaxTrainSentLength = 110;

    }
}
