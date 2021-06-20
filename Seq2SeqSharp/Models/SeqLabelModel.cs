using Seq2SeqSharp.Models;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    [Serializable]
    public class SeqLabelModel : Model
    {
        public SeqLabelModel()
        {

        }

        public SeqLabelModel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab clsVocab)
                        : base(hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, false)
        {
            ClsVocab = clsVocab;
        }
    }
}
