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

        public int m_hdim { get; set; }
        public int m_dim { get; set; }

        private int m_batchSize;
        private int m_deviceId;

        private LayerNormalization layerNorm1;
        private LayerNormalization layerNorm2;

        public LSTMCell(int batchSize, int hdim, int dim, ArchTypeEnums archType, int deviceId)
        {
            Wxh = new WeightTensor(dim + hdim, hdim * 4, deviceId);
            b = new WeightTensor(1, hdim * 4, 0, deviceId);

            m_hdim = hdim;
            m_dim = dim;
            m_batchSize = batchSize;
            m_deviceId = deviceId;

            layerNorm1 = new LayerNormalization(hdim * 4, archType, deviceId);
            layerNorm2 = new LayerNormalization(hdim, archType, deviceId);
        }

        public IWeightMatrix Step(IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;
        
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var bs = innerGraph.RepeatRows(b, input.Rows);
            var hhSum = innerGraph.MulAdd(inputs, Wxh, bs);
            var hhSum2 = layerNorm1.Process(hhSum, innerGraph);

            (var gates_raw, var cell_write_raw) = innerGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = innerGraph.Sigmoid(gates_raw);
            var cell_write = innerGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = innerGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            ct = innerGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(ct, innerGraph);

            // compute hidden state as gated, saturated cell activations
            ht = innerGraph.EltMul(output_gate, innerGraph.Tanh(ct2));

            return ht;
        }

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
            response.Add(Wxh);
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
            Wxh.Save(stream);
            b.Save(stream);

            layerNorm1.Save(stream);
            layerNorm2.Save(stream);

        }


        public void Load(Stream stream)
        {
            Wxh.Load(stream);
            b.Load(stream);

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
        }
    }
     
}
