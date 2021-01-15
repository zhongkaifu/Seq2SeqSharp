using System;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int SrcEmbeddingDim;
        public int TgtEmbeddingDim;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public DecoderTypeEnums DecoderType;
        public Vocab Vocab;
        public bool EnableCoverageModel = true;

        public Seq2SeqModelMetaData()
        {

        }

        public Seq2SeqModelMetaData(int hiddenDim, int srcEmbeddingDim, int tgtEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab vocab, bool enableCoverageModel)
        {
            HiddenDim = hiddenDim;
            SrcEmbeddingDim = srcEmbeddingDim;
            TgtEmbeddingDim = tgtEmbeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            DecoderType = decoderType;
            Vocab = vocab;
            EnableCoverageModel = enableCoverageModel;
        }
    }
}
