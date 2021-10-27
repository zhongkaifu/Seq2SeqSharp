using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public class SeqSimilarityModel : Model
    {
        public SeqSimilarityModel()
        {

        }

        public SeqSimilarityModel(int hiddenDim, int embeddingDim, int encoderLayerDepth, int multiHeadNum, EncoderTypeEnums encoderType, Vocab srcVocab, Vocab clsVocab, bool enableSegmentEmbeddings, string similarityType, int maxSegmentNum)
                        : base(hiddenDim, encoderLayerDepth, encoderType, embeddingDim, multiHeadNum, srcVocab, enableSegmentEmbeddings, false, maxSegmentNum)
        {
            ClsVocab = clsVocab;
            SimilarityType = similarityType;
        }
    }
}
