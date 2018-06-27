
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{

    [Serializable]
    public class Encoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public Encoder()
        {

        }

        public Encoder(int hdim, int dim, int depth )
        {
            encoders.Add(new LSTMCell(hdim, dim));

            //for (int i = 1; i < depth; i++)
            //{
            //   encoders.Add(new LSTMCell(hdim, hdim));

            //}
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }
        public void Reset()
        {
            foreach (var item in encoders)
            {
                item.Reset();
            }

        }

        public WeightMatrix Encode(WeightMatrix V, IComputeGraph g)
        {
            foreach (var encoder in encoders)
            {
                var e = encoder.Step(V, g); 
                    V = e; 
  
            }
            return V;
        }
        public List<WeightMatrix> Encode2(WeightMatrix V, IComputeGraph g)
        {
            List<WeightMatrix> res = new List<WeightMatrix>();
            foreach (var encoder in encoders)
            {
                var e = encoder.Step(V, g);
                V = e;
                res.Add(e);
            }
            return res;
        }

        public List<WeightMatrix> getParams()
        {
            List<WeightMatrix> response = new List<WeightMatrix>();

            foreach (var item in encoders)
            {
                response.AddRange(item.getParams());

            }



            return response;
        }

    }
}
