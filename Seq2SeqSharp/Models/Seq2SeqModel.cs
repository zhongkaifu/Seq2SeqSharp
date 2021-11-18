using System;
using System.Linq;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModel : Model
    {
        public Seq2SeqModel() { }

        public Seq2SeqModel( int hiddenDim, int encoderEmbeddingDim, int decoderEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum,
                             EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, bool enableCoverageModel,
                             bool sharedEmbeddings, bool enableSegmentEmbeddings, bool applyContextEmbeddingsToEntireSequence, int maxSegmentNum )
            : base( hiddenDim, encoderLayerDepth, encoderType, encoderEmbeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings, applyContextEmbeddingsToEntireSequence, maxSegmentNum )
        {
            DecoderEmbeddingDim = decoderEmbeddingDim;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            DecoderType = decoderType;
            EnableCoverageModel = enableCoverageModel;
            SharedEmbeddings = sharedEmbeddings;
            TgtVocab = tgtVocab;
        }

        public Seq2SeqModel( Model_4_ProtoBufSerializer m )
            : base( m.HiddenDim, m.EncoderLayerDepth, m.EncoderType, m.EncoderEmbeddingDim, m.MultiHeadNum,
                    m.SrcVocab?.ToVocab(),
                    m.EnableSegmentEmbeddings, m.ApplyContextEmbeddingsToEntireSequence, m.MaxSegmentNum )
        {
            ClsVocabs    = m.ClsVocabs?.Select( v => v.ToVocab() ).ToList();
            Name2Weights = m.Name2Weights;

            DecoderEmbeddingDim = m.DecoderEmbeddingDim;
            DecoderLayerDepth   = m.DecoderLayerDepth;
            DecoderType         = m.DecoderType;
            EnableCoverageModel = m.EnableCoverageModel;
            SharedEmbeddings    = m.SharedEmbeddings;
            TgtVocab            = m.TgtVocab?.ToVocab();
        }
        public static Seq2SeqModel Create( Model_4_ProtoBufSerializer m ) => new Seq2SeqModel( m );
    }
}
