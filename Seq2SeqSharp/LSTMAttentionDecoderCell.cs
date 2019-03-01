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
        private int m_deviceId;

        public LSTMAttentionDecoderCell(int batchSize, int hdim, int dim, ArchTypeEnums archType, int deviceId, bool isDefaultDevice)
        {
            int contextSize = hdim * 2;
            this.hdim = hdim;
            this.dim = dim;
            m_deviceId = deviceId;

            m_batchSize = batchSize;

            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                Wxhc = new WeightTensor(dim + hdim + contextSize, hdim * 4, deviceId, isDefaultDevice, true);
                b = new WeightTensor(1, hdim * 4, 0, deviceId, isDefaultDevice);

                this.ht = new WeightTensor(batchSize, hdim, 0, deviceId, isDefaultDevice);
                this.ct = new WeightTensor(batchSize, hdim, 0, deviceId, isDefaultDevice);
            }
            else
            {
                Wxhc = new WeightMatrix(dim + hdim + contextSize, hdim * 4, true);
                b = new WeightMatrix(1, hdim * 4, 0);

                this.ht = new WeightMatrix(batchSize, hdim, 0);
                this.ct = new WeightMatrix(batchSize, hdim, 0);
            }
        }

        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public IWeightMatrix Step(IWeightMatrix context, IWeightMatrix input, IComputeGraph computeGraph)
        {
            var cell_prev = ct;
            var hidden_prev = ht;

            var hxhc = computeGraph.ConcatColumns(input, hidden_prev, context);
            var bs = computeGraph.RepeatRows(b, input.Rows);
            var hhSum = computeGraph.MulAdd(hxhc, Wxhc, bs);

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum, hdim * 3, hdim);
            var gates = computeGraph.Sigmoid(gates_raw, true);
            var cell_write = computeGraph.Tanh(cell_write_raw, true);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, hdim, hdim, hdim);

            // compute new cell activation
            //var retain_cell = computeGraph.EltMul(forget_gate, cell_prev);
            //var write_cell = computeGraph.EltMul(input_gate, cell_write);

            //ct = computeGraph.Add(retain_cell, write_cell);


            ct = computeGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);

            ht = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct));

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
            ht = weightFactory.CreateWeights(m_batchSize, hdim, m_deviceId, true);
            ct = weightFactory.CreateWeights(m_batchSize, hdim, m_deviceId, true);
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


