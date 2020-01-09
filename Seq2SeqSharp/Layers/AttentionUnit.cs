using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{

    public class AttentionPreProcessResult
    {
        public IWeightTensor rawInputs;
        public IWeightTensor uhs;
        public IWeightTensor inputs;

    }

    [Serializable]
    public class AttentionUnit : INeuralUnit
    {
        private readonly IWeightTensor m_V;
        private readonly IWeightTensor m_Ua;
        private readonly IWeightTensor m_bUa;
        private readonly IWeightTensor m_Wa;
        private readonly IWeightTensor m_bWa;

        private readonly string m_name;
        private readonly int m_hiddenDim;
        private readonly int m_contextDim;
        private readonly int m_deviceId;

        private bool m_enableCoverageModel = true;
        private readonly IWeightTensor m_Wc;
        private readonly IWeightTensor m_bWc;
        private readonly LSTMCell m_coverage;

        public AttentionUnit(string name, int hiddenDim, int contextDim, int deviceId)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_contextDim = contextDim;
            m_deviceId = deviceId;

            Logger.WriteLine($"Creating attention unit '{name}' HiddenDim = '{hiddenDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}'");

            m_Ua = new WeightTensor(new long[2] { contextDim, hiddenDim }, deviceId, normal: true, name: $"{name}.{nameof(m_Ua)}", isTrainable: true);
            m_Wa = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, normal: true, name: $"{name}.{nameof(m_Wa)}", isTrainable: true);
            m_bUa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bUa)}", isTrainable: true);
            m_bWa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bWa)}", isTrainable: true);
            m_V = new WeightTensor(new long[2] { hiddenDim, 1 }, deviceId, normal: true, name: $"{name}.{nameof(m_V)}", isTrainable: true);

            if (m_enableCoverageModel)
            {
                m_Wc = new WeightTensor(new long[2] { 16, hiddenDim }, deviceId, normal: true, name: $"{name}.{nameof(m_Wc)}", isTrainable: true);
                m_bWc = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bWc)}", isTrainable: true);
                m_coverage = new LSTMCell(name: $"{name}.{nameof(m_coverage)}", hdim: 16, dim: 1 + contextDim + hiddenDim, deviceId: deviceId);
            }
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor inputs, int batchSize, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name + "_PreProcess");
            AttentionPreProcessResult r = new AttentionPreProcessResult
            {
                rawInputs = inputs,
                uhs = g.Affine(inputs, m_Ua, m_bUa),
                inputs = g.TransposeBatch(inputs, batchSize)
            };

            if (m_enableCoverageModel)
            {
                m_coverage.Reset(graph.GetWeightFactory(), r.inputs.Rows);
            }

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attenPreProcessResult, int batchSize, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name);

            IWeightTensor wc = g.Affine(state, m_Wa, m_bWa);
            IWeightTensor wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / batchSize);
            IWeightTensor wcSum = wcs;

            if (m_enableCoverageModel)
            {
                IWeightTensor wCoverage = g.Affine(m_coverage.Hidden, m_Wc, m_bWc);
                wcSum = g.Add(wcs, wCoverage);
            }

            IWeightTensor ggs = g.AddTanh(attenPreProcessResult.uhs, wcSum);
            IWeightTensor atten = g.Mul(ggs, m_V);

            IWeightTensor atten2 = g.TransposeBatch(atten, batchSize);
            IWeightTensor attenT = g.Transpose(atten2);
            IWeightTensor attenT2 = g.View(attenT, batchSize, attenPreProcessResult.inputs.Rows / batchSize);

            IWeightTensor attenSoftmax1 = g.Softmax(attenT2, inPlace: true);

            IWeightTensor attenSoftmax = g.View(attenSoftmax1, batchSize, attenSoftmax1.Rows / batchSize, attenSoftmax1.Columns);
            IWeightTensor inputs2 = g.View(attenPreProcessResult.inputs, batchSize, attenPreProcessResult.inputs.Rows / batchSize, attenPreProcessResult.inputs.Columns);

            IWeightTensor contexts = g.MulBatch(attenSoftmax, inputs2, batchSize);

            if (m_enableCoverageModel)
            {
                // Concatenate tensor as input for coverage model
                IWeightTensor aCoverage = g.View(attenSoftmax1, attenPreProcessResult.inputs.Rows, 1);
                IWeightTensor aCoverage2 = g.TransposeBatch(aCoverage, attenPreProcessResult.inputs.Rows / batchSize);
                IWeightTensor sCoverage = g.RepeatRows(state, attenPreProcessResult.inputs.Rows / batchSize);
                IWeightTensor concate = g.ConcatColumns(aCoverage2, attenPreProcessResult.rawInputs, sCoverage);
                m_coverage.Step(concate, graph);
            }

            return contexts;
        }


        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_Ua,
                m_Wa,
                m_bUa,
                m_bWa,
                m_V
            };

            if (m_enableCoverageModel)
            {
                response.Add(m_Wc);
                response.Add(m_bWc);
                response.AddRange(m_coverage.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            m_Ua.Save(stream);
            m_Wa.Save(stream);
            m_bUa.Save(stream);
            m_bWa.Save(stream);
            m_V.Save(stream);

            if (m_enableCoverageModel)
            {
                m_Wc.Save(stream);
                m_bWc.Save(stream);
                m_coverage.Save(stream);
            }
        }


        public void Load(Stream stream)
        {
            m_Ua.Load(stream);
            m_Wa.Load(stream);
            m_bUa.Load(stream);
            m_bWa.Load(stream);
            m_V.Load(stream);

            if (m_enableCoverageModel)
            {
                m_Wc.Load(stream);
                m_bWc.Load(stream);
                m_coverage.Load(stream);
            }
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            AttentionUnit a = new AttentionUnit(m_name, m_hiddenDim, m_contextDim, deviceId);
            return a;
        }
    }
}



