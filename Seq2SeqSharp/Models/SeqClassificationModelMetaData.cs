using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    class SeqClassificationModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab Vocab;
        public bool EnableSegmentEmbeddings = false;

        public SeqClassificationModelMetaData()
        {

        }

        public SeqClassificationModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab vocab, bool enableSegmentEmbeddings)
        {
            HiddenDim = hiddenDim;
            EmbeddingDim = embeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            Vocab = vocab;
            EnableSegmentEmbeddings = enableSegmentEmbeddings;
        }
    }
}
