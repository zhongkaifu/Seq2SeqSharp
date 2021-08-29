using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModel : Model
    {
        public Seq2SeqModel()
        {
        }

        public Seq2SeqModel(int hiddenDim, int encoderEmbeddingDim, int decoderEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, 
            EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, bool enableCoverageModel, bool sharedEmbeddings, bool enableSegmentEmbeddings, bool applyContextEmbeddingsToEntireSequence)
            :base(hiddenDim, encoderLayerDepth, encoderType, encoderEmbeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings, applyContextEmbeddingsToEntireSequence)
        {
            DecoderEmbeddingDim = decoderEmbeddingDim;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            DecoderType = decoderType;
            EnableCoverageModel = enableCoverageModel;
            SharedEmbeddings = sharedEmbeddings;
            TgtVocab = tgtVocab;
        }
    }
}
