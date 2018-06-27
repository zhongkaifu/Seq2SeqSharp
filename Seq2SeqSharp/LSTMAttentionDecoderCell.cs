using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{


    [Serializable]
    public class LSTMAttentionDecoderCell : LSTMCell 
    {
        public WeightMatrix WiC { get; set; }
        public WeightMatrix WfC { get; set; }
        public WeightMatrix WoC { get; set; }

        public WeightMatrix WcC { get; set; }

        public WeightMatrix WiS { get; set; }
        public WeightMatrix WfS { get; set; }
        public WeightMatrix WoS { get; set; }

        public WeightMatrix WcS { get; set; }

        public int sdim { get; set; }

        public LSTMAttentionDecoderCell()
        {

        }

        public LSTMAttentionDecoderCell(int sdim, int hdim, int dim)
            : base(hdim, dim)
        {
            int contextSize = hdim * 2;
            this.WiC = new WeightMatrix(contextSize, hdim, true);
            this.WfC = new WeightMatrix(contextSize, hdim, true);
            this.WoC = new WeightMatrix(contextSize, hdim, true);
            this.WcC = new WeightMatrix(contextSize, hdim, true);

            this.sdim = sdim;

            if (sdim > 0)
            {
                this.WiS = new WeightMatrix(sdim, hdim, true);
                this.WfS = new WeightMatrix(sdim, hdim, true);
                this.WoS = new WeightMatrix(sdim, hdim, true);
                this.WcS = new WeightMatrix(sdim, hdim, true);
            }
        }

        public WeightMatrix Step(SparseWeightMatrix sparseInput, WeightMatrix context, WeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;

            var cell = this;
            WeightMatrix input_gate = null;
            WeightMatrix forget_gate = null;
            WeightMatrix output_gate = null;
            WeightMatrix cell_write = null;

            Parallel.Invoke(
                () =>
                {

                    var h0 = innerGraph.mul(input, cell.Wix);
                    var h1 = innerGraph.mul(hidden_prev, cell.Wih);
                    var h11 = innerGraph.mul(context, cell.WiC);

                    if (sdim > 0)
                    {
                        var h111 = innerGraph.mul(sparseInput, cell.WiS);
                        input_gate = innerGraph.addsigmoid(h0, h1, h11, h111, cell.bi);
                    }
                    else
                    {
                        input_gate = innerGraph.addsigmoid(h0, h1, h11, cell.bi);
                    }
                },
                () =>
                {

                    var h2 = innerGraph.mul(input, cell.Wfx);
                    var h3 = innerGraph.mul(hidden_prev, cell.Wfh);
                    var h33 = innerGraph.mul(context, cell.WfC);

                    if (sdim > 0)
                    {
                        var h333 = innerGraph.mul(sparseInput, cell.WfS);
                        forget_gate = innerGraph.addsigmoid(h3, h2, h33, h333, cell.bf);
                    }
                    else
                    {
                        forget_gate = innerGraph.addsigmoid(h3, h2, h33, cell.bf);
                    }
                },
                () =>
                {

                    var h4 = innerGraph.mul(input, cell.Wox);
                    var h5 = innerGraph.mul(hidden_prev, cell.Woh);
                    var h55 = innerGraph.mul(context, cell.WoC);

                    if (sdim > 0)
                    {
                        var h555 = innerGraph.mul(sparseInput, cell.WoS);
                        output_gate = innerGraph.addsigmoid(h5, h4, h55, h555, cell.bo);
                    }
                    else
                    {
                        output_gate = innerGraph.addsigmoid(h5, h4, h55, cell.bo);
                    }
                },
                () =>
                {

                    var h6 = innerGraph.mul(input, cell.Wcx);
                    var h7 = innerGraph.mul(hidden_prev, cell.Wch);
                    var h77 = innerGraph.mul(context, cell.WcC);

                    if (sdim > 0)
                    {
                        var h777 = innerGraph.mul(sparseInput, cell.WcS);
                        cell_write = innerGraph.addtanh(h7, h6, h77, h777, cell.bc);
                    }
                    else
                    {
                        cell_write = innerGraph.addtanh(h7, h6, h77, cell.bc);
                    }
                });

            // compute new cell activation
            var retain_cell = innerGraph.eltmul(forget_gate, cell_prev); // what do we keep from cell
            var write_cell = innerGraph.eltmul(input_gate, cell_write); // what do we write to cell
            var cell_d = innerGraph.add(retain_cell, write_cell); // new cell contents

            // compute hidden state as gated, saturated cell activations
            var hidden_d = innerGraph.eltmul(output_gate, innerGraph.tanh(cell_d));

            this.ht = hidden_d;
            this.ct = cell_d;

            return ht;
        }
       
        public override List<WeightMatrix> getParams()
        {
            List<WeightMatrix> response = new List<WeightMatrix>();
            response.AddRange(base.getParams()); 
            
            response.Add(this.WiC);
            response.Add(this.WfC);
            response.Add(this.WoC);
            response.Add(this.WcC);

            if (sdim > 0)
            {
                response.Add(this.WiS);
                response.Add(this.WfS);
                response.Add(this.WoS);
                response.Add(this.WcS);
            }

            return response;
        }
         

    }


}
