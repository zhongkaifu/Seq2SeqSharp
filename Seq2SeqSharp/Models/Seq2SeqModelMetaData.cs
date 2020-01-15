using System;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab Vocab;
        public bool EnableCoverageModel = true;

        public Seq2SeqModelMetaData()
        {

        }

        public Seq2SeqModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab vocab, bool enableCoverageModel)
        {
            HiddenDim = hiddenDim;
            EmbeddingDim = embeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            Vocab = vocab;
            EnableCoverageModel = enableCoverageModel;
        }
    }
}
