using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public abstract class Model
    {
        public int HiddenDim;
        public int EncoderLayerDepth;
        public EncoderTypeEnums EncoderType;
        public int MultiHeadNum;
        public Vocab SrcVocab;

        public Dictionary<string, float[]> Name2Weights { get; set; }

        public Model()
        {

        }

        public Model(int hiddenDim, int encoderLayerDepth, EncoderTypeEnums encoderType, int multiHeadNum, Vocab srcVocab)
        {
            HiddenDim = hiddenDim;
            EncoderLayerDepth = encoderLayerDepth;
            EncoderType = encoderType;
            MultiHeadNum = multiHeadNum;
            SrcVocab = srcVocab;

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
