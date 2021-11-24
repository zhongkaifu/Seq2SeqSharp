using System;
using System.Linq;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp
{
    [Serializable]
    public class SeqLabelModel : Model
    {
        public SeqLabelModel() { }
        public SeqLabelModel( int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab clsVocab, int maxSegmentNum)
            : base( hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, false, false, maxSegmentNum, false )
        {
            ClsVocab = clsVocab;
        }
        public SeqLabelModel( Model_4_ProtoBufSerializer m )
            : base( m.HiddenDim, m.EncoderLayerDepth, m.EncoderType, m.EncoderEmbeddingDim, m.MultiHeadNum,
                    m.SrcVocab?.ToVocab(), 
                    enableSegmentEmbeddings: false, applyContextEmbeddingsToEntireSequence: false, m.MaxSegmentNum, false )
        {
            ClsVocabs    = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList(); 
            Name2Weights = m.Name2Weights;
        }
        public static SeqLabelModel Create( Model_4_ProtoBufSerializer m ) => new SeqLabelModel( m );
    }
}
