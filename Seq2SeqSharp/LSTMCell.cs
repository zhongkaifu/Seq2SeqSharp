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
        int m_batchSize;
        int m_deviceId;

        LayerNormalization m_layerNorm1;
        LayerNormalization m_layerNorm2;

        public LSTMCell(int batchSize, int hdim, int dim, int deviceId)
        {
            m_Wxh = new WeightTensor(dim + hdim, hdim * 4, deviceId, true);
            m_b = new WeightTensor(1, hdim * 4, 0, deviceId);

            m_hdim = hdim;
            m_dim = dim;
            m_batchSize = batchSize;
            m_deviceId = deviceId;

            m_layerNorm1 = new LayerNormalization(hdim * 4, deviceId);
            m_layerNorm2 = new LayerNormalization(hdim, deviceId);
        }

        public IWeightTensor Step(IWeightTensor input, IComputeGraph innerGraph)
        {
            var hidden_prev = m_hidden;
            var cell_prev = m_cell;
        
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var bs = innerGraph.RepeatRows(m_b, input.Rows);
            var hhSum = innerGraph.MulAdd(inputs, m_Wxh, bs);
            var hhSum2 = m_layerNorm1.Process(hhSum, innerGraph);

            (var gates_raw, var cell_write_raw) = innerGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = innerGraph.Sigmoid(gates_raw);
            var cell_write = innerGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = innerGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            m_cell = innerGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = m_layerNorm2.Process(m_cell, innerGraph);

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

        public void Reset(IWeightFactory weightFactory)
        {
            m_hidden = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true);
            m_cell = weightFactory.CreateWeights(m_batchSize, m_hdim, m_deviceId, true);
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
