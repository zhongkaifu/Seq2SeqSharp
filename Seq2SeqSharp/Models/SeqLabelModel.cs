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
    public class SeqLabelModel : Model, IModel
    {
        public int EmbeddingDim;
        public Vocab TgtVocab;

        public SeqLabelModel()
        {

        }

        public SeqLabelModel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab tgtVocab)
                        : base(hiddenDim, encoderLayerDepth, encoderType, multiHeadNum, srcVocab)
        {
            EmbeddingDim = embeddingDim;
            TgtVocab = tgtVocab;
        }
    }
}
