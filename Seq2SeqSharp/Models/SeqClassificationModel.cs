using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    class SeqClassificationModel : Model, IModel
    {
        public int EmbeddingDim;
        public bool EnableSegmentEmbeddings = false;
        public List<Vocab> TgtVocabs;

        public SeqClassificationModel()
        {

        }

        public SeqClassificationModel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, List<Vocab> tgtVocabs, bool enableSegmentEmbeddings)
                        : base(hiddenDim, encoderLayerDepth, encoderType, multiHeadNum, srcVocab)
        {
            TgtVocabs = tgtVocabs;
            EmbeddingDim = embeddingDim;
            EnableSegmentEmbeddings = enableSegmentEmbeddings;
        }
    }
}
