using System;
using System.Collections.Generic;
using System.Linq;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class SeqClassificationModel : Model
    {
        public SeqClassificationModel() { }
        public SeqClassificationModel( int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, List<Vocab> clsVocabs, bool enableSegmentEmbeddings, bool applyContextEmbeddingsToEntireSequence, int maxSegmentNum )
            : base( hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings, applyContextEmbeddingsToEntireSequence, maxSegmentNum )
        {
            ClsVocabs = clsVocabs;
        }
        public SeqClassificationModel( Model_4_ProtoBufSerializer m )
            : base( m.HiddenDim, m.EncoderLayerDepth, m.EncoderType, m.EncoderEmbeddingDim, m.MultiHeadNum,
                    m.SrcVocab?.ToVocab(),
                    m.EnableSegmentEmbeddings, m.ApplyContextEmbeddingsToEntireSequence, m.MaxSegmentNum )
        {
            ClsVocabs    = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList();
            Name2Weights = m.Name2Weights;
        }
        public static SeqClassificationModel Create( Model_4_ProtoBufSerializer m ) => new SeqClassificationModel( m );
    }
}
