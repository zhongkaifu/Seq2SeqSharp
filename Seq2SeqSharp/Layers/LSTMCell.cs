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
        IWeightTensor m_Wxh;
        IWeightTensor m_b;
        IWeightTensor m_hidden;
        IWeightTensor m_cell;

        int m_hdim;
        int m_dim;
        int m_deviceId;
        string m_name;

        LayerNormalization m_layerNorm1;
        LayerNormalization m_layerNorm2;

        public LSTMCell(string name, int hdim, int dim, int deviceId)
        {
            m_name = name;

            m_Wxh = new WeightTensor(new long[2] { dim + hdim, hdim * 4 }, deviceId, normal: true, name: $"{name}.{nameof(m_Wxh)}", isTrainable: true);
            m_b = new WeightTensor(new long[2] { 1, hdim * 4 }, 0, deviceId, name: $"{name}.{nameof(m_b)}", isTrainable: true);

            m_hdim = hdim;
            m_dim = dim;
            m_deviceId = deviceId;

            m_layerNorm1 = new LayerNormalization($"{name}.{nameof(m_layerNorm1)}", hdim * 4, deviceId);
            m_layerNorm2 = new LayerNormalization($"{name}.{nameof(m_layerNorm2)}", hdim, deviceId);
        }

        public IWeightTensor Step(IWeightTensor input, IComputeGraph g)
        {
            var innerGraph = g.CreateSubGraph(m_name);

            var hidden_prev = m_hidden;
            var cell_prev = m_cell;
        
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var hhSum = innerGraph.Affine(inputs, m_Wxh, m_b);
            var hhSum2 = m_layerNorm1.Norm(hhSum, innerGraph);

            (var gates_raw, var cell_write_raw) = innerGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = innerGraph.Sigmoid(gates_raw);
            var cell_write = innerGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = innerGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            m_cell = innerGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = m_layerNorm2.Norm(m_cell, innerGraph);

            // compute hidden state as gated, saturated cell activations
            m_hidden = innerGraph.EltMul(output_gate, innerGraph.Tanh(ct2));

            return m_hidden;
        }

        public virtual List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();
            response.Add(m_Wxh);
            response.Add(m_b);

            response.AddRange(m_layerNorm1.getParams());
            response.AddRange(m_layerNorm2.getParams());

            return response;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            if (m_hidden != null)
            {               
                m_hidden.Dispose();
                m_hidden = null;
            }

            if (m_cell != null)
            {
                m_cell.Dispose();
                m_cell = null;
            }

            m_hidden = weightFactory.CreateWeightTensor(batchSize, m_hdim, m_deviceId, true, name: $"{m_name}.{nameof(m_hidden)}", isTrainable: true);
            m_cell = weightFactory.CreateWeightTensor(batchSize, m_hdim, m_deviceId, true, name: $"{m_name}.{nameof(m_cell)}", isTrainable: true);
        }

        public void Save(Stream stream)
        {
            m_Wxh.Save(stream);
            m_b.Save(stream);

            m_layerNorm1.Save(stream);
            m_layerNorm2.Save(stream);

        }


        public void Load(Stream stream)
        {
            m_Wxh.Load(stream);
            m_b.Load(stream);

            m_layerNorm1.Load(stream);
            m_layerNorm2.Load(stream);
        }
    }
     
}
