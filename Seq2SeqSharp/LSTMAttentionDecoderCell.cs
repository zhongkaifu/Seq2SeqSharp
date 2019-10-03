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
        public IWeightTensor Hidden { get; set; }
        public IWeightTensor Cell { get; set; }

        int m_hdim;
        int m_dim;
        int m_batchSize;
        int m_deviceId;
        string m_name;

        IWeightTensor m_Wxhc;
        IWeightTensor m_b;

        LayerNormalization layerNorm1;
        LayerNormalization layerNorm2;

        public LSTMAttentionDecoderCell(string name, int batchSize, int hdim, int dim, int contextSize, int deviceId)
        {
            m_name = name;
            m_hdim = hdim;
            m_dim = dim;
            m_deviceId = deviceId;
            m_batchSize = batchSize;

            m_Wxhc = new WeightTensor(new long[2] { dim + hdim + contextSize, hdim * 4 }, deviceId, normal: true, name: $"{name}.{nameof(m_Wxhc)}", isTrainable: true);
            m_b = new WeightTensor(new long[2] { 1, hdim * 4 }, 0, deviceId, name: $"{name}.{nameof(m_b)}", isTrainable: true);

            Hidden = new WeightTensor(new long[2] { batchSize, hdim }, 0, deviceId, name: $"{name}.{nameof(Hidden)}", isTrainable: true);
            Cell = new WeightTensor(new long[2] { batchSize, hdim }, 0, deviceId, name: $"{name}.{nameof(Cell)}", isTrainable: true);

            layerNorm1 = new LayerNormalization($"{name}.{nameof(layerNorm1)}", hdim * 4, deviceId);
            layerNorm2 = new LayerNormalization($"{name}.{nameof(layerNorm2)}", hdim, deviceId);
        }

        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public IWeightTensor Step(IWeightTensor context, IWeightTensor input, IComputeGraph g)
        {
            var computeGraph = g.CreateSubGraph(m_name);

            var cell_prev = Cell;
            var hidden_prev = Hidden;

            var hxhc = computeGraph.ConcatColumns(input, hidden_prev, context);
            var hhSum = computeGraph.Affine(hxhc, m_Wxhc, m_b);
            var hhSum2 = layerNorm1.Process(hhSum, computeGraph);

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = computeGraph.Sigmoid(gates_raw);
            var cell_write = computeGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            Cell = computeGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(Cell, computeGraph);

            Hidden = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct2));

            return Hidden;
        }

        public List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();
            response.Add(m_Wxhc);
            response.Add(m_b);

            response.AddRange(layerNorm1.getParams());
            response.AddRange(layerNorm2.getParams());

            return response;
        }

        public void Reset(IWeightFactory weightFactory)
        {
            Hidden = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true, name: $"{m_name}.{nameof(Hidden)}", isTrainable: true);
            Cell = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true, name: $"{m_name}.{nameof(Cell)}", isTrainable: true);
        }

        public void Save(Stream stream)
        {
            m_Wxhc.Save(stream);
            m_b.Save(stream);

            layerNorm1.Save(stream);
            layerNorm2.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_Wxhc.Load(stream);
            m_b.Load(stream);

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
        }
    }
}


