using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{
    [Serializable]
    public class LSTMAttentionDecoderCell
    {
        public IWeightMatrix Wxhc { get; set; }

        public IWeightMatrix b { get; set; }

        public IWeightMatrix ht { get; set; }
        public IWeightMatrix ct { get; set; }

        public int hdim { get; set; }
        public int dim { get; set; }

        public int m_batchSize;

        public LSTMAttentionDecoderCell(int batchSize, int hdim, int dim, ArchTypeEnums archType)
        {
            int contextSize = hdim * 2;
            this.hdim = hdim;
            this.dim = dim;

            m_batchSize = batchSize;

            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                Wxhc = new WeightTensor(dim + hdim + contextSize, hdim * 4, true);
                b = new WeightTensor(1, hdim * 4, 0);

                this.ht = new WeightTensor(batchSize, hdim, 0);
                this.ct = new WeightTensor(batchSize, hdim, 0);
            }
            else
            {
                Wxhc = new WeightMatrix(dim + hdim + contextSize, hdim * 4, true);
                b = new WeightMatrix(1, hdim * 4, 0);

                this.ht = new WeightMatrix(batchSize, hdim, 0);
                this.ct = new WeightMatrix(batchSize, hdim, 0);
            }
        }

        public IWeightMatrix Step(IWeightMatrix context, IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;

            var cell = this;
            IWeightMatrix input_gate = null;
            IWeightMatrix forget_gate = null;
            IWeightMatrix output_gate = null;
            IWeightMatrix cell_write = null;

            var bs = innerGraph.RepeatRows(b, input.Rows);
            var hxhc = innerGraph.ConcatColumns(new IWeightMatrix[] { input, hidden_prev, context });
            var hhSum = innerGraph.MulAdd(hxhc, Wxhc, bs);
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
            response.Add(Wxhc);
            response.Add(b);

            return response;
        }


        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            m_batchSize = batchSize;
            Reset(weightFactory);
        }

        public void Reset(IWeightFactory weightFactory)
        {
            ht = weightFactory.CreateWeights(m_batchSize, hdim, true);
            ct = weightFactory.CreateWeights(m_batchSize, hdim, true);
        }

        public void Save(Stream stream)
        {
            Wxhc.Save(stream);
            b.Save(stream);
        }


        public void Load(Stream stream)
        {
            Wxhc.Load(stream);
            b.Load(stream);
        }
    }
}


