using Seq2SeqSharp.Utils;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IModel
    {
        public int DecoderEmbeddingDim { get; set; }
        public int EncoderEmbeddingDim { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int EncoderLayerDepth { get; set; }
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
        public Vocab ClsVocab { get; set; }


        public void AddWeights(string name, float[] weights);

        public float[] GetWeights(string name);

        public void ClearWeights();
    }
}
