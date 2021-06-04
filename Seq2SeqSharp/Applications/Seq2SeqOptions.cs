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

        [Arg("Decoder type: AttentionLSTM, Transformer", "DecoderType")]
        public string DecoderType = "Transformer";

        [Arg("Apply coverage model in decoder", "EnableCoverageModel")]
        public bool EnableCoverageModel = true;

        [Arg("It indicates if the decoder is trainable", "IsDecoderTrainable")]
        public bool IsDecoderTrainable = true;

        [Arg("It indicates if the encoder is trainable", "IsEncoderTrainable")]
        public bool IsEncoderTrainable = true;

        [Arg("It indicates if the src embedding is trainable", "IsSrcEmbeddingTrainable")]
        public bool IsSrcEmbeddingTrainable = true;
        [Arg("It indicates if the tgt embedding is trainable", "IsTgtEmbeddingTrainable")]
        public bool IsTgtEmbeddingTrainable = true;

        [Arg("Maxmium src sentence length in valid and test set", "MaxSrcTestSentLength")]
        public int MaxSrcTestSentLength = 32;

        [Arg("Maxmium src sentence length in training corpus", "MaxSrcTrainSentLength")]
        public int MaxSrcTrainSentLength = 110;

        [Arg("Maxmium tgt sentence length in valid and test set", "MaxTgtTestSentLength")]
        public int MaxTgtTestSentLength = 32;

        [Arg("Maxmium tgt sentence length in training corpus", "MaxTgtTrainSentLength")]
        public int MaxTgtTrainSentLength = 110;

        [Arg("It indicates if output alignment between target tokens and source tokens", "OutputAlignment")]
        public bool OutputAlignment = false;

        [Arg("Sharing embeddings between source side and target side", "SharedEmbeddings")]
        public bool SharedEmbeddings = false;

        [Arg("The embedding dim in source side", "SrcEmbeddingDim")]
        public int SrcEmbeddingDim = 128;

        [Arg("The external embedding model file path for source side.", "SrcEmbedding")]
        public string SrcEmbeddingModelFilePath = null;

        [Arg("Source language name.", "SrcLang")]
        public string SrcLang;

        [Arg("The embedding dim in target side", "TgtEmbeddingDim")]
        public int TgtEmbeddingDim = 128;

        [Arg("The external embedding model file path for target side.", "TgtEmbedding")]
        public string TgtEmbeddingModelFilePath = null;

        [Arg("Target language name.", "TgtLang")]
        public string TgtLang;
    }
}
