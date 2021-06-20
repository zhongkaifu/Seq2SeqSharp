using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    class SeqClassificationModel : Model
    {
        public SeqClassificationModel()
        {

        }

        public SeqClassificationModel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, List<Vocab> clsVocabs, bool enableSegmentEmbeddings)
                        : base(hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings)
        {
            ClsVocabs = clsVocabs;
        }
    }
}
