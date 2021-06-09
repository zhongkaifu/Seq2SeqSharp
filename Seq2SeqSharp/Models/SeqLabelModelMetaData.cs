using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    [Serializable]
    public class SeqLabelModelMetaData : IModelMetaData
    {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab Vocab;

        public Dictionary<string, float[]> Name2Weights { get; set; }
        public SeqLabelModelMetaData()
        {

        }

        public SeqLabelModelMetaData(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab vocab)
        {
            HiddenDim = hiddenDim;
            EmbeddingDim = embeddingDim;
            EncoderLayerDepth = encoderLayerDepth;
            MultiHeadNum = multiHeadNum;
            EncoderType = encoderType;
            Vocab = vocab;

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
