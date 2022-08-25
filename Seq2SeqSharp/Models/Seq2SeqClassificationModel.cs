using System;
using System.Linq;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class Seq2SeqClassificationModel : Seq2SeqModel
    {

        public Seq2SeqClassificationModel() { }
        public Seq2SeqClassificationModel(int hiddenDim, int srcEmbeddingDim, int tgtEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum,
            EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, Vocab clsVocab, bool enableCoverageModel, bool sharedEmbeddings, bool enableSegmentEmbeddings, bool enableTagEmbeddings, int maxSegmentNum, int expertNum)
            : base(hiddenDim, srcEmbeddingDim, tgtEmbeddingDim, encoderLayerDepth, decoderLayerDepth, multiHeadNum, encoderType, decoderType, srcVocab, tgtVocab, enableCoverageModel, sharedEmbeddings, enableSegmentEmbeddings, 
                  enableTagEmbeddings, maxSegmentNum, pointerGenerator: false, expertNum: expertNum)
        {
            ClsVocab = clsVocab;
        }
        public Seq2SeqClassificationModel(Model_4_ProtoBufSerializer m)
            : base(m.HiddenDim, m.EncoderEmbeddingDim, m.DecoderEmbeddingDim, m.EncoderLayerDepth, m.DecoderLayerDepth, m.MultiHeadNum,
                    m.EncoderType, m.DecoderType, m.SrcVocab?.ToVocab(), m.TgtVocab?.ToVocab(), m.EnableCoverageModel, m.SharedEmbeddings,
                    m.EnableSegmentEmbeddings, m.EnableTagEmbeddings, m.MaxSegmentNum, pointerGenerator: false, expertNum: m.ExpertNum)
        {
            ClsVocabs = m.ClsVocabs?.Select(v => v.ToVocab()).ToList();
            Name2Weights = m.Name2Weights;
        }
        public static new Seq2SeqClassificationModel Create(Model_4_ProtoBufSerializer m) => new Seq2SeqClassificationModel(m);
    }
}
