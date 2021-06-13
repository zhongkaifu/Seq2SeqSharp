using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModel : Model, IModel
    {

        public int SrcEmbeddingDim;
        public int TgtEmbeddingDim;
        public int DecoderLayerDepth;
        public DecoderTypeEnums DecoderType;
        public bool EnableCoverageModel = true;
        public bool SharedEmbeddings = false;
        public bool EnableSegmentEmbeddings = false;
        public Vocab TgtVocab;
        public Seq2SeqModel()
        {
        }

        public Seq2SeqModel(int hiddenDim, int srcEmbeddingDim, int tgtEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, 
            EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, bool enableCoverageModel, bool sharedEmbeddings, bool enableSegmentEmbeddings)
            :base(hiddenDim, encoderLayerDepth, encoderType, multiHeadNum, srcVocab)
        {
            SrcEmbeddingDim = srcEmbeddingDim;
            TgtEmbeddingDim = tgtEmbeddingDim;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            DecoderType = decoderType;
            EnableCoverageModel = enableCoverageModel;
            SharedEmbeddings = sharedEmbeddings;
            EnableSegmentEmbeddings = enableSegmentEmbeddings;
            TgtVocab = tgtVocab;
        }
    }
}
