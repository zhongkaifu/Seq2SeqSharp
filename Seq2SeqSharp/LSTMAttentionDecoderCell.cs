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
    public class LSTMAttentionDecoderCell //: LSTMCell 
    {
#if CUDA

    //    public IWeightMatrix W { get; set; }
        public IWeightMatrix Wxhc { get; set; }
#endif

        public IWeightMatrix Wx { get; set; }
        public IWeightMatrix Wh { get; set; }

        public IWeightMatrix Wc { get; set; }

        public IWeightMatrix b { get; set; }

        public IWeightMatrix WiS { get; set; }
        public IWeightMatrix WfS { get; set; }
        public IWeightMatrix WoS { get; set; }

        public IWeightMatrix WcS { get; set; }

        public IWeightMatrix ht { get; set; }
        public IWeightMatrix ct { get; set; }

        public int hdim { get; set; }
        public int dim { get; set; }

        public int sdim { get; set; }

        public LSTMAttentionDecoderCell(int sdim, int hdim, int dim)
         //   : base(hdim, dim)
        {
            int contextSize = hdim * 2;
            this.sdim = sdim;
            this.hdim = hdim;
            this.dim = dim;

#if CUDA
            //ComputeGraphTensor g = new ComputeGraphTensor(false);
            //W = new WeightTensor(dim + hdim + contextSize + 1, hdim * 4, true);
            //List<IWeightMatrix> Ws = g.SplitRows(W, dim + hdim + contextSize, 1);
            ////Wx = Ws[0];
            ////Wh = Ws[1];

            //Wxhc = Ws[0];
            //b = Ws[1];

            Wxhc = new WeightTensor(dim + hdim + contextSize, hdim * 4, true);
            b = new WeightTensor(1, hdim * 4, 0);

            //Wx = new WeightTensor(dim, hdim * 4, true);
            //Wh = new WeightTensor(hdim, hdim * 4, true);
            //b = new WeightTensor(1, hdim * 4, true);

            this.ht = new WeightTensor(1, hdim, 0);
            this.ct = new WeightTensor(1, hdim, 0);

       //     Wc = new WeightTensor(contextSize, hdim * 4, true);

#else
          
            Wc = new WeightMatrix(contextSize, hdim * 4, true);

            Wx = new WeightMatrix(dim, hdim * 4, true);
            Wh = new WeightMatrix(hdim, hdim * 4, true);
            b = new WeightMatrix(1, hdim * 4, true);

            this.ht = new WeightMatrix(1, hdim, 0);
            this.ct = new WeightMatrix(1, hdim, 0);
#endif

            if (sdim > 0)
            {
#if CUDA
                this.WiS = new WeightTensor(sdim, hdim, true);
                this.WfS = new WeightTensor(sdim, hdim, true);
                this.WoS = new WeightTensor(sdim, hdim, true);
                this.WcS = new WeightTensor(sdim, hdim, true);
#else
                this.WiS = new WeightMatrix(sdim, hdim, true);
                this.WfS = new WeightMatrix(sdim, hdim, true);
                this.WoS = new WeightMatrix(sdim, hdim, true);
                this.WcS = new WeightMatrix(sdim, hdim, true);
#endif
            }
        }

        public IWeightMatrix Step(SparseWeightMatrix sparseInput, IWeightMatrix context, IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;

            var cell = this;
            IWeightMatrix input_gate = null;
            IWeightMatrix forget_gate = null;
            IWeightMatrix output_gate = null;
            IWeightMatrix cell_write = null;

#if CUDA
            var hxhc = innerGraph.ConcatColumns(new IWeightMatrix[] { input, hidden_prev, context });
            var hhSum = innerGraph.MulAdd(hxhc, Wxhc, b);
#else
            var hx = innerGraph.Mul(input, Wx);
            var hh = innerGraph.Mul(hidden_prev, Wh);
            var hc = innerGraph.Mul(context, Wc);
            var hhSum = innerGraph.Add(hx, hh, hc, b);
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
       
        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

#if CUDA

            response.Add(Wxhc);
            response.Add(b);
#else
            response.Add(Wx);
            response.Add(Wh);
            response.Add(Wc);
            response.Add(b);
#endif

            if (sdim > 0)
            {
                response.Add(this.WiS);
                response.Add(this.WfS);
                response.Add(this.WoS);
                response.Add(this.WcS);
            }

            return response;
        }

        //public override List<float[]> GetWeightList()
        //{
        //    List<float[]> wl = new List<float[]>();
        //    wl.AddRange(base.GetWeightList());

        //    wl.Add(Wc.ToWeightArray());

        //    if (sdim > 0)
        //    {
        //        wl.Add(WiS.ToWeightArray());
        //        wl.Add(WfS.ToWeightArray());
        //        wl.Add(WoS.ToWeightArray());
        //        wl.Add(WcS.ToWeightArray());

        //    }

        //    return wl;
        //}

        //public override void SetWeightList(List<float[]> wl)
        //{
        //    base.SetWeightList(wl);

        //    Wc.SetWeightArray(wl[0]);

        //    wl.RemoveRange(0, 1);


        //    if (sdim > 0)
        //    {
        //        WiS.SetWeightArray(wl[0]);
        //        WfS.SetWeightArray(wl[1]);
        //        WoS.SetWeightArray(wl[2]);
        //        WcS.SetWeightArray(wl[3]);

        //        wl.RemoveRange(0, 4);
        //    }

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
