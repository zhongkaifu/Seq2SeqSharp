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
    public class LSTMCell 
    {
#if CUDA

      //  public IWeightMatrix W { get; set; }
        public IWeightMatrix Wxh { get; set; }
#endif


        public IWeightMatrix Wx { get; set; }
        public IWeightMatrix Wh { get; set; }
        public IWeightMatrix b { get; set; }

        public IWeightMatrix ht { get; set; }
        public IWeightMatrix ct { get; set; }


        public int hdim { get; set; }
        public int dim { get; set; }

        public LSTMCell(int hdim, int dim)
        {
#if CUDA
            //ComputeGraphTensor g = new ComputeGraphTensor(false);
            //W = new WeightTensor(dim + hdim + 1, hdim * 4, true);
            //List<IWeightMatrix> Ws = g.SplitRows(W, dim + hdim, 1);
            ////Wx = Ws[0];
            ////Wh = Ws[1];

            //Wxh = Ws[0];
            //b = Ws[1];

            Wxh = new WeightTensor(dim + hdim, hdim * 4, true);
            b = new WeightTensor(1, hdim * 4, 0);

            //Wx = new WeightTensor(dim, hdim * 4, true);
            //Wh = new WeightTensor(hdim, hdim * 4, true);
            //b = new WeightTensor(1, hdim * 4, true);

            this.ht = new WeightTensor(1, hdim, 0);
            this.ct = new WeightTensor(1, hdim, 0);
#else

            Wx = new WeightMatrix(dim, hdim * 4, true);
            Wh = new WeightMatrix(hdim, hdim * 4, true);
            b = new WeightMatrix(1, hdim * 4, true);

            this.ht = new WeightMatrix(1, hdim, 0);
            this.ct = new WeightMatrix(1, hdim, 0);

#endif
            this.hdim = hdim;
            this.dim = dim;
        }

        public IWeightMatrix Step(IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;
            var cell = this;
            IWeightMatrix input_gate = null, forget_gate = null, output_gate = null, cell_write = null;

#if CUDA
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var hhSum = innerGraph.MulAdd(inputs, Wxh, b);
#else
            var hx = innerGraph.Mul(input, Wx);
            var hh = innerGraph.Mul(hidden_prev, Wh);
            var hhSum = innerGraph.Add(hx, hh, b);
#endif
            var paramList = innerGraph.SplitColumns(hhSum, hdim * 3, hdim);

            var gates = innerGraph.Sigmoid(paramList[0]);
            cell_write = innerGraph.Tanh(paramList[1]);

            var gateList = innerGraph.SplitColumns(gates, hdim, hdim, hdim);
            input_gate = gateList[0];
            forget_gate = gateList[1];
            output_gate = gateList[2];


            // compute new cell activation
            var retain_cell = innerGraph.EltMul(forget_gate, cell_prev); // what do we keep from cell
            var write_cell = innerGraph.EltMul(input_gate, cell_write); // what do we write to cell
            var cell_d = innerGraph.Add(retain_cell, write_cell); // new cell contents

            // compute hidden state as gated, saturated cell activations
            var hidden_d = innerGraph.EltMul(output_gate, innerGraph.Tanh(cell_d));

            this.ht = hidden_d;
            this.ct = cell_d;
            return ht;
        }

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
#if CUDA
            response.Add(Wxh);
            response.Add(b);
#else
            response.Add(Wx);
            response.Add(Wh);
            response.Add(b);
#endif
            return response;
        }

        //public virtual List<float[]> GetWeightList()
        //{
        //    List<float[]> weightList = new List<float[]>();

        //    weightList.Add(Wx.ToWeightArray());
        //    weightList.Add(Wh.ToWeightArray());
        //    weightList.Add(b.ToWeightArray());

        //    return weightList;

        //}

        //public virtual void SetWeightList(List<float[]> wl)
        //{        
        //    Wx.SetWeightArray(wl[0]);
        //    Wh.SetWeightArray(wl[1]);
        //    b.SetWeightArray(wl[2]);

        //    wl.RemoveRange(0, 3);

        //}


        public void Reset()
        {
            ht.ClearWeight();
            ct.ClearWeight();

            ht.ClearGradient();
            ct.ClearGradient();
        }

    }
     
}
