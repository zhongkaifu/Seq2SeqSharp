using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace Seq2SeqSharp
{

    public class AttentionPreProcessResult
    {
        public IWeightTensor Uhs;
        public IWeightTensor encOutput;
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
        private readonly bool m_isTrainable;

        private bool m_enableCoverageModel;
        private readonly IWeightTensor m_Wc;
        private readonly IWeightTensor m_bWc;
        private readonly LSTMCell m_coverage;

        private readonly int k_coverageModelDim = 16;

        public AttentionUnit(string name, int hiddenDim, int contextDim, int deviceId, bool enableCoverageModel, bool isTrainable)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_contextDim = contextDim;
            m_deviceId = deviceId;
            m_enableCoverageModel = enableCoverageModel;
            m_isTrainable = isTrainable;

            Logger.WriteLine($"Creating attention unit '{name}' HiddenDim = '{hiddenDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}', EnableCoverageModel = '{enableCoverageModel}'");

            m_Ua = new WeightTensor(new long[2] { contextDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(m_Ua)}", isTrainable: isTrainable);
            m_Wa = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(m_Wa)}", isTrainable: isTrainable);
            m_bUa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bUa)}", isTrainable: isTrainable);
            m_bWa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bWa)}", isTrainable: isTrainable);
            m_V = new WeightTensor(new long[2] { hiddenDim, 1 }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(m_V)}", isTrainable: isTrainable);

            if (m_enableCoverageModel)
            {
                m_Wc = new WeightTensor(new long[2] { k_coverageModelDim, hiddenDim }, deviceId, normal: NormType.Uniform, name: $"{name}.{nameof(m_Wc)}", isTrainable: isTrainable);
                m_bWc = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bWc)}", isTrainable: isTrainable);
                m_coverage = new LSTMCell(name: $"{name}.{nameof(m_coverage)}", hdim: k_coverageModelDim, dim: 1 + contextDim + hiddenDim, deviceId: deviceId, isTrainable: isTrainable);
            }
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor encOutput, int batchSize, IComputeGraph g)
        {
            int srcSeqLen = encOutput.Rows / batchSize;

            AttentionPreProcessResult r = new AttentionPreProcessResult
            {
                encOutput = encOutput
            };

            r.Uhs = g.Affine(r.encOutput, m_Ua, m_bUa);
            r.Uhs = g.View(r.Uhs, dims: new long[] { batchSize, srcSeqLen, -1 });


            if (m_enableCoverageModel)
            {
                m_coverage.Reset(g.GetWeightFactory(), r.encOutput.Rows);
            }

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attnPre, int batchSize, IComputeGraph graph)
        {
            int srcSeqLen = attnPre.encOutput.Rows / batchSize;

            using (IComputeGraph g = graph.CreateSubGraph(m_name))
            {
                // Affine decoder state
                IWeightTensor wc = g.Affine(state, m_Wa, m_bWa);

                // Expand dims from [batchSize x decoder_dim] to [batchSize x srcSeqLen x decoder_dim]
                IWeightTensor wc1 = g.View(wc, dims: new long[] { batchSize, 1, wc.Columns });
                IWeightTensor wcExp = g.Expand(wc1, dims: new long[] { batchSize, srcSeqLen, wc.Columns });

                IWeightTensor ggs = null;
                if (m_enableCoverageModel)
                {
                    // Get coverage model status at {t-1}
                    IWeightTensor wCoverage = g.Affine(m_coverage.Hidden, m_Wc, m_bWc);
                    IWeightTensor wCoverage1 = g.View(wCoverage, dims: new long[] { batchSize, srcSeqLen, -1 });

                    ggs = g.AddTanh(attnPre.Uhs, wcExp, wCoverage1);
                }
                else
                {
                    ggs = g.AddTanh(attnPre.Uhs, wcExp);
                }

                IWeightTensor ggss = g.View(ggs, dims: new long[] { batchSize * srcSeqLen, -1 });
                IWeightTensor atten = g.Mul(ggss, m_V);

                IWeightTensor attenT = g.Transpose(atten);
                IWeightTensor attenT2 = g.View(attenT, dims: new long[] { batchSize, srcSeqLen });

                IWeightTensor attenSoftmax1 = g.Softmax(attenT2, inPlace: true);

                IWeightTensor attenSoftmax = g.View(attenSoftmax1, dims: new long[] { batchSize, 1, srcSeqLen });
                IWeightTensor inputs2 = g.View(attnPre.encOutput, dims: new long[] { batchSize, srcSeqLen, attnPre.encOutput.Columns });

                IWeightTensor contexts = graph.MulBatch(attenSoftmax, inputs2, batchSize);

                if (m_enableCoverageModel)
                {
                    // Concatenate tensor as input for coverage model
                    IWeightTensor aCoverage = g.View(attenSoftmax1, dims: new long[] { attnPre.encOutput.Rows, 1 });


                    IWeightTensor state2 = g.View(state, dims: new long[] { batchSize, 1, state.Columns });
                    IWeightTensor state3 = g.Expand(state2, dims: new long[] { batchSize, srcSeqLen, state.Columns });
                    IWeightTensor state4 = g.View(state3, dims: new long[] { batchSize * srcSeqLen, -1 });


                    IWeightTensor concate = g.ConcatColumns(aCoverage, attnPre.encOutput, state4);
                    m_coverage.Step(concate, graph);
                }


                return contexts;
            }
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
            AttentionUnit a = new AttentionUnit(m_name, m_hiddenDim, m_contextDim, deviceId, m_enableCoverageModel, m_isTrainable);
            return a;
        }
    }
}



