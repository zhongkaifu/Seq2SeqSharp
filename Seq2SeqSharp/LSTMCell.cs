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
    public class LSTMCell 
    {
        public IWeightMatrix Wxh { get; set; }

        public IWeightMatrix b { get; set; }

        public IWeightMatrix ht { get; set; }
        public IWeightMatrix ct { get; set; }


        public int hdim { get; set; }
        public int dim { get; set; }

        private int batchSize;
        private int deviceId;

        public LSTMCell(int batchSize, int hdim, int dim, ArchTypeEnums archType, int deviceId)
        {
            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                Wxh = new WeightTensor(dim + hdim, hdim * 4, deviceId, true);
                b = new WeightTensor(1, hdim * 4, 0, deviceId);
            }
            else
            {
                Wxh = new WeightMatrix(dim + hdim, hdim * 4, true);
                b = new WeightMatrix(1, hdim * 4, 0);
            }

            this.hdim = hdim;
            this.dim = dim;
            this.batchSize = batchSize;
            this.deviceId = deviceId;
        }

        public IWeightMatrix Step(IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;
            var cell = this;
            IWeightMatrix input_gate = null, forget_gate = null, output_gate = null, cell_write = null;

            var bs = innerGraph.RepeatRows(b, input.Rows);
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var hhSum = innerGraph.MulAdd(inputs, Wxh, bs);
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
            response.Add(Wxh);
            response.Add(b);

            return response;
        }

        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            this.batchSize = batchSize;
            Reset(weightFactory);
        }

        public void Reset(IWeightFactory weightFactory)
        {
            ht = weightFactory.CreateWeights(batchSize, hdim, deviceId, true);
            ct = weightFactory.CreateWeights(batchSize, hdim, deviceId, true);
        }

        public void Save(Stream stream)
        {
            Wxh.Save(stream);
            b.Save(stream);
        }


        public void Load(Stream stream)
        {
            Wxh.Load(stream);
            b.Load(stream);
        }
    }
     
}
