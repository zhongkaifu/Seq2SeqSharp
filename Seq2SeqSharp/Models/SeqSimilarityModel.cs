using System;
using System.Linq;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class SeqSimilarityModel : Model
    {
        public SeqSimilarityModel() { }
        public SeqSimilarityModel( int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab clsVocab, bool enableSegmentEmbeddings, string similarityType, int maxSegmentNum )
            : base( hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings, false, maxSegmentNum )
        {
            ClsVocab       = clsVocab;
            SimilarityType = similarityType;
        }
        public SeqSimilarityModel( Model_4_ProtoBufSerializer m )
            : base( m.HiddenDim, m.EncoderLayerDepth, m.EncoderType, m.EncoderEmbeddingDim, m.MultiHeadNum,
                    m.SrcVocab?.ToVocab(),
                    m.EnableSegmentEmbeddings, applyContextEmbeddingsToEntireSequence: false, m.MaxSegmentNum )
        {
            ClsVocabs    = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList();
            Name2Weights = m.Name2Weights;
        }
        public static SeqSimilarityModel Create( Model_4_ProtoBufSerializer m ) => new SeqSimilarityModel( m );
    }
}
