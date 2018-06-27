
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{


    [Serializable]
    public class ContextDecoder
    {
        public List<LSTMContextDecoderCell> decoders = new List<LSTMContextDecoderCell>(); 
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public ContextDecoder(int hdim, int dim, int depth)
        {
             decoders.Add(new LSTMContextDecoderCell(hdim, dim));
             for (int i = 1; i < depth; i++)
             {
                 decoders.Add(new LSTMContextDecoderCell(hdim, hdim));
  
             }
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }
        public void Reset()
        {
            foreach (var item in decoders)
            {
                item.Reset();
            }

        } 
        public WeightMatrix Decode(WeightMatrix input, WeightMatrix encoderOutput, IComputeGraph g)
        {
            var V = new WeightMatrix();
            
            foreach (var encoder in decoders)
            { 
                var e = encoder.Step(encoderOutput, input, g);
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
            
            return response;
        }

    }
}
