
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{


    [Serializable]
    public class AttentionDecoder
    {
        public List<LSTMAttentionDecoderCell> decoders = new List<LSTMAttentionDecoderCell>(); 
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }
        public AttentionUnit Attention { get; set; }

        public AttentionDecoder()
        {

        }

        public AttentionDecoder(int sdim, int hdim, int dim, int depth)
        {
            Attention = new AttentionUnit(hdim);
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;

            decoders.Add(new LSTMAttentionDecoderCell(sdim, hdim, dim));
            for (int i = 1; i < depth; i++)
            {
                decoders.Add(new LSTMAttentionDecoderCell(0, hdim, hdim));
            }
        }
        public void Reset()
        {
            foreach (var item in decoders)
            {
                item.Reset();
            }

        }
        public WeightMatrix Decode(SparseWeightMatrix sparseInput, WeightMatrix input, List<WeightMatrix> encoderOutput, IComputeGraph g)
        {
            var V = input;
            var lastStatus = this.decoders.FirstOrDefault().ct;
            var context = Attention.Perform(encoderOutput, lastStatus, g);
            foreach (var decoder in decoders)
            {
                var e = decoder.Step(sparseInput, context, V, g);
                V = e;
            }

            return V;
        } 
       
        public WeightMatrix Decode(SparseWeightMatrix sparseInput, WeightMatrix input, WeightMatrix encoderOutput, IComputeGraph g)
        {
            var V = input;
            var lastStatus = this.decoders.FirstOrDefault().ct;
            var context = Attention.Perform(encoderOutput, lastStatus, g);
            foreach (var decoder in decoders)
            {
                var e = decoder.Step(sparseInput, context, V, g);
                V = e; 
            }

            return V;
        }

        public List<WeightMatrix> getParams()
        {
            List<WeightMatrix> response = new List<WeightMatrix>();

            foreach (var item in decoders)
            {
                response.AddRange(item.getParams());
            }
            response.AddRange(Attention.getParams());
            return response;
        }

    }
}
