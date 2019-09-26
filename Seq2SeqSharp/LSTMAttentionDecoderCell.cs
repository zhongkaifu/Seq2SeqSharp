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

        public int m_hdim { get; set; }
        public int m_dim { get; set; }

        public int m_batchSize;
        private int m_deviceId;

        private LayerNormalization layerNorm1;
        private LayerNormalization layerNorm2;

        public LSTMAttentionDecoderCell(int batchSize, int hdim, int dim, int contextSize, ArchTypeEnums archType, int deviceId)
        {
            m_hdim = hdim;
            m_dim = dim;
            m_deviceId = deviceId;
            m_batchSize = batchSize;

            Wxhc = new WeightTensor(dim + hdim + contextSize, hdim * 4, deviceId);
            b = new WeightTensor(1, hdim * 4, 0, deviceId);

            ht = new WeightTensor(batchSize, hdim, 0, deviceId);
            ct = new WeightTensor(batchSize, hdim, 0, deviceId);

            layerNorm1 = new LayerNormalization(hdim * 4, archType, deviceId);
            layerNorm2 = new LayerNormalization(hdim, archType, deviceId);
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
            var hhSum2 = layerNorm1.Process(hhSum, computeGraph);

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = computeGraph.Sigmoid(gates_raw);
            var cell_write = computeGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            ct = computeGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(ct, computeGraph);

            ht = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct2));

            return ht;
        }

        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
            response.Add(Wxhc);
            response.Add(b);

            response.AddRange(layerNorm1.getParams());
            response.AddRange(layerNorm2.getParams());

            return response;
        }

        public void Reset(IWeightFactory weightFactory)
        {
            ht = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true);
            ct = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true);
        }

        public void Save(Stream stream)
        {
            Wxhc.Save(stream);
            b.Save(stream);

            layerNorm1.Save(stream);
            layerNorm2.Save(stream);
        }


        public void Load(Stream stream)
        {
            Wxhc.Load(stream);
            b.Load(stream);

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
        }
    }
}


