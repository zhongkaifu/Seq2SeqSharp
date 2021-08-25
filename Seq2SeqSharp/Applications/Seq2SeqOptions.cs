using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AdvUtils;
using Seq2SeqSharp.Applications;

namespace Seq2SeqSharp
{
    public class Seq2SeqOptions : Options
    {
        [Arg("The network depth in decoder.", "DecoderLayerDepth")]
        public int DecoderLayerDepth = 1;

        [Arg("Starting Learning rate factor for decoders", "DecoderStartLearningRateFactor")]
        public float DecoderStartLearningRateFactor = 1.0f;

        [Arg("Decoder type: AttentionLSTM, Transformer", "DecoderType")]
        public string DecoderType = "Transformer";

        [Arg("Apply coverage model in decoder", "EnableCoverageModel")]
        public bool EnableCoverageModel = true;

        [Arg("It indicates if the decoder is trainable", "IsDecoderTrainable")]
        public bool IsDecoderTrainable = true;



        [Arg("It indicates if the src embedding is trainable", "IsSrcEmbeddingTrainable")]
        public bool IsSrcEmbeddingTrainable = true;
        [Arg("It indicates if the tgt embedding is trainable", "IsTgtEmbeddingTrainable")]
        public bool IsTgtEmbeddingTrainable = true;

        [Arg("Maxmium src sentence length in valid and test set", "MaxTestSrcSentLength")]
        public int MaxTestSrcSentLength = 32;

        [Arg("Maxmium tgt sentence length in valid and test set", "MaxTestTgtSentLength")]
        public int MaxTestTgtSentLength = 32;

        [Arg("Maxmium src sentence length in training corpus", "MaxTrainSrcSentLength")]
        public int MaxTrainSrcSentLength = 110;

        [Arg("Maxmium tgt sentence length in training corpus", "MaxTrainTgtSentLength")]
        public int MaxTrainTgtSentLength = 110;

        [Arg("The metric for sequence generation task. It supports BLEU and RougeL", "SeqGenerationMetric")]
        public string SeqGenerationMetric = "BLEU";

        [Arg("Sharing embeddings between source side and target side", "SharedEmbeddings")]
        public bool SharedEmbeddings = false;

        [Arg("The embedding dim in source side", "SrcEmbeddingDim")]
        public int SrcEmbeddingDim = 128;

        [Arg("The embedding dim in target side", "TgtEmbeddingDim")]
        public int TgtEmbeddingDim = 128;

    }
}
