using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    [Serializable]
    public class Seq2SeqModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int SrcEmbeddingDim;
        public int TgtEmbeddingDim;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public DecoderTypeEnums DecoderType;
        public Vocab SrcVocab;
        public Vocab TgtVocab;
        public bool EnableCoverageModel = true;
        public bool SharedEmbeddings = false;
        public bool EnableSegmentEmbeddings = false;

        public Dictionary<string, float[]> Name2Weights { get; set; }

        public Seq2SeqModelMetaData()
        {

        }

        public Seq2SeqModelMetaData(int hiddenDim, int srcEmbeddingDim, int tgtEmbeddingDim, int encoderLayerDepth, int decoderLayerDepth, int multiHeadNum, 
            EncoderTypeEnums encoderType, DecoderTypeEnums decoderType, Vocab srcVocab, Vocab tgtVocab, bool enableCoverageModel, bool sharedEmbeddings, bool enableSegmentEmbeddings)
        {
            HiddenDim = hiddenDim;
            SrcEmbeddingDim = srcEmbeddingDim;
            TgtEmbeddingDim = tgtEmbeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            DecoderLayerDepth = decoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            DecoderType = decoderType;
            SrcVocab = srcVocab;
            TgtVocab = tgtVocab;
            EnableCoverageModel = enableCoverageModel;
            SharedEmbeddings = sharedEmbeddings;
            EnableSegmentEmbeddings = enableSegmentEmbeddings;

            Name2Weights = new Dictionary<string, float[]>();
        }

        public void AddWeights(string name, float[] weights)
        {
            Name2Weights.Add(name, weights);
        }

        public float[] GetWeights(string name)
        {
            return Name2Weights[name];
        }

        public void ClearWeights()
        {
            Name2Weights.Clear();
        }
    }
}
