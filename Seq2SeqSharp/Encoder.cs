
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{

    [Serializable]
    public class Encoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public Encoder(int hdim, int dim, int depth)
        {
            encoders.Add(new LSTMCell(hdim, dim));

            for (int i = 1; i < depth; i++)
            {
                encoders.Add(new LSTMCell(hdim, hdim));

            }
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

        public IWeightMatrix Encode(IWeightMatrix V, IComputeGraph g)
        {
            foreach (var encoder in encoders)
            {
                var e = encoder.Step(V, g);
                V = e;
            }

            return V;
        }


        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in encoders)
            {
                response.AddRange(item.getParams());

            }

            return response;
        }

        //public List<float[]> GetWeightList()
        //{
        //    List<float[]> wl = new List<float[]>();

        //    foreach (var item in encoders)
        //    {
        //        wl.AddRange(item.GetWeightList());
        //    }

        //    return wl;
        //}

        //public void SetWeightList(List<float[]> wl)
        //{
        //    foreach (var item in encoders)
        //    {
        //        item.SetWeightList(wl);
        //    }
        //}
    }
}
