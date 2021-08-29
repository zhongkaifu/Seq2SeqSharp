using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class Seq2SeqClassificationModel : Seq2SeqModel
    {
 
        public Seq2SeqClassificationModel()
        {
        }

        public Seq2SeqClassificationModel(int hiddenDim, int srcEmbeddingDim, int tgtEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum,
            EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, Vocab clsVocab, bool enableCoverageModel, bool sharedEmbeddings, bool enableSegmentEmbeddings, bool applyContextEmbeddingsToEntireSequence)
            : base(hiddenDim, srcEmbeddingDim, tgtEmbeddingDim, encoderLayerDepth, decoderLayerDepth, multiHeadNum, encoderType, decoderType, srcVocab, tgtVocab, enableCoverageModel, sharedEmbeddings, enableSegmentEmbeddings, applyContextEmbeddingsToEntireSequence)
        {
            ClsVocab = clsVocab;
        }
    }
}
