using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    class SeqClassificationModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab SrcVocab;
        public Vocab TgtVocab;
        public bool EnableSegmentEmbeddings = false;

        public Dictionary<string, float[]> Name2Weights { get; set; }
        public SeqClassificationModelMetaData()
        {

        }

        public SeqClassificationModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab tgtVocab, bool enableSegmentEmbeddings)
        {
            HiddenDim = hiddenDim;
            EmbeddingDim = embeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            SrcVocab = srcVocab;
            TgtVocab = tgtVocab;
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
