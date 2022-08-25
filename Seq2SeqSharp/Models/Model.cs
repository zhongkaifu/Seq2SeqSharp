using System;
using System.Collections.Generic;

using AdvUtils;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public abstract class Model : IModel
    {
        public int DecoderEmbeddingDim { get; set; }
        public int EncoderEmbeddingDim { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int EncoderLayerDepth { get; set; }

        public int ExpertNum { get; set; }
        public DecoderTypeEnums DecoderType { get; set; }
        public EncoderTypeEnums EncoderType { get; set; }
        public int HiddenDim { get; set; }
        public bool EnableSegmentEmbeddings { get; set; }
        public int MultiHeadNum { get; set; }
        public Vocab SrcVocab { get; set; }
        public Vocab TgtVocab { get; set; }
        public List<Vocab> ClsVocabs { get; set; }
        public bool EnableCoverageModel { get; set; }
        public bool SharedEmbeddings { get; set; }

        public string SimilarityType { get; set; }

        public bool EnableTagEmbeddings { get; set; }

        public int MaxSegmentNum { get; set; }

        public bool PointerGenerator { get; set; }

        public Vocab ClsVocab
        {
            get
            {
                if ( ClsVocabs == null )
                {
                    ClsVocabs = new List<Vocab>
                    {
                        new Vocab()
                    };
                }

                return ClsVocabs[ 0 ];
            }

            set
            {
                if ( ClsVocabs == null )
                {
                    ClsVocabs = new List<Vocab>
                    {
                        new Vocab()
                    };
                }

                ClsVocabs[ 0 ] = value;
            }
        }


        public Dictionary<string, float[]> Name2Weights { get; set; }

        public Model() { }
        public Model( int hiddenDim, int encoderLayerDepth, EncoderTypeEnums encoderType, int encoderEmbeddingDim, int multiHeadNum, Vocab srcVocab,
            bool enableSegmentEmbeddings, bool enableTagEmbeddings, int maxSegmentNum, bool pointerGenerator, int expertNum )
        {
            HiddenDim = hiddenDim;
            EncoderLayerDepth = encoderLayerDepth;
            EncoderType = encoderType;
            MultiHeadNum = multiHeadNum;
            SrcVocab = srcVocab;
            EncoderEmbeddingDim = encoderEmbeddingDim;
            EnableSegmentEmbeddings = enableSegmentEmbeddings;
            EnableTagEmbeddings = enableTagEmbeddings;
            MaxSegmentNum = maxSegmentNum;
            PointerGenerator = pointerGenerator;
            ExpertNum = expertNum;

            Name2Weights = new Dictionary<string, float[]>();
        }

        public void AddWeights( string name, float[] weights )
        {
            Name2Weights.Add( name, weights );
        }

        public float[] GetWeights( string name )
        {
            if ( Name2Weights.ContainsKey( name ) == false )
            {
                Logger.WriteLine( Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model." );
                return null;
            }

            return Name2Weights[ name ];
        }

        public void ClearWeights()
        {
            Name2Weights.Clear();
        }

        public void ShowModelInfo()
        {
            Logger.WriteLine( $"Encoder embedding dim: '{EncoderEmbeddingDim}'" );
            Logger.WriteLine( $"Decoder embedding dim: '{DecoderEmbeddingDim}'" );
            Logger.WriteLine( $"Encoder layer depth: '{EncoderLayerDepth}'" );
            Logger.WriteLine( $"Decoder layer depth: '{DecoderLayerDepth}'" );
            Logger.WriteLine( $"Encoder type: '{EncoderType}'" );
            Logger.WriteLine( $"Decoder type: '{DecoderType}'" );
            Logger.WriteLine( $"Hidden layer dim: '{HiddenDim}'" );
            Logger.WriteLine( $"Enable segment embeddings: '{EnableSegmentEmbeddings}'" );
            Logger.WriteLine( $"Enable shared embeddings: '{SharedEmbeddings}'" );
            Logger.WriteLine( $"Enable tag embeddings: '{EnableTagEmbeddings}'" );
            Logger.WriteLine( $"Multi-head size: '{MultiHeadNum}'" );
            Logger.WriteLine($"Pointer Generator: '{PointerGenerator}'");
            Logger.WriteLine($"Expert Num: '{ExpertNum}");


            if ( ! SimilarityType.IsNullOrEmpty() )
            {
                Logger.WriteLine( $"Similarity Type: '{SimilarityType}'" );
            }

            if ( SrcVocab != null )
            {
                Logger.WriteLine( $"Source vocabulary size: '{SrcVocab.Count}'" );
            }

            if ( TgtVocab != null )
            {
                Logger.WriteLine( $"Target vocabulary size: '{TgtVocab.Count}'" );
            }

            if ( ClsVocabs != null )
            {
                Logger.WriteLine( $"The number of CLS vocabularies: '{ClsVocabs.Count}' " );
                for ( int i = 0; i < ClsVocabs.Count; i++ )
                {
                    Logger.WriteLine( $"CLS vocabulary {i} size: {ClsVocabs[ i ].Count}" );
                }
            }
        }
    }
}
