using AdvUtils;
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

    public class AttentionPreProcessResult
    {
        public IWeightTensor uhs;
        public IWeightTensor inputs;

    }

    [Serializable]
    public class AttentionUnit : INeuralUnit
    {
        IWeightTensor m_V;
        IWeightTensor m_Ua;
        IWeightTensor m_bUa;
        IWeightTensor m_Wa;
        IWeightTensor m_bWa;

        string m_name;
        int m_hiddenDim;
        int m_contextDim;
        int m_deviceId;

        public AttentionUnit(string name, int hiddenDim, int contextDim, int deviceId)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_contextDim = contextDim;
            m_deviceId = deviceId;

            Logger.WriteLine($"Creating attention unit '{name}' HiddenDim = '{hiddenDim}', ContextDim = '{contextDim}', DeviceId = '{deviceId}'");

            m_Ua = new WeightTensor(new long[2] { contextDim, hiddenDim }, deviceId, normal: true, name: $"{name}.{nameof(m_Ua)}", isTrainable: true);
            m_Wa = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, normal:true, name: $"{name}.{nameof(m_Wa)}", isTrainable: true);
            m_bUa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bUa)}", isTrainable: true);
            m_bWa = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(m_bWa)}", isTrainable: true);
            m_V = new WeightTensor(new long[2] { hiddenDim, 1 }, deviceId, normal:true, name: $"{name}.{nameof(m_V)}", isTrainable: true);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor inputs, int batchSize, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name + "_PreProcess");
            AttentionPreProcessResult r = new AttentionPreProcessResult();

            r.uhs = g.Affine(inputs, m_Ua, m_bUa);
            r.inputs = g.TransposeBatch(inputs, batchSize);

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attenPreProcessResult, int batchSize, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name);

            var wc = g.Affine(state, m_Wa, m_bWa);
            var wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / batchSize);
            var ggs = g.AddTanh(attenPreProcessResult.uhs, wcs);
            var atten = g.Mul(ggs, m_V);

            var atten2 = g.TransposeBatch(atten, batchSize);
            var attenT = g.Transpose(atten2);
            var attenT2 = g.View(attenT, batchSize, attenPreProcessResult.inputs.Rows / batchSize);

            var attenSoftmax1 = g.Softmax(attenT2, inPlace: true);

            var attenSoftmax = g.View(attenSoftmax1, batchSize, attenSoftmax1.Rows / batchSize, attenSoftmax1.Columns);
            var inputs2 = g.View(attenPreProcessResult.inputs, batchSize, attenPreProcessResult.inputs.Rows / batchSize, attenPreProcessResult.inputs.Columns);

            IWeightTensor contexts = g.MulBatch(attenSoftmax, inputs2, batchSize);

            return contexts;
        }

     
        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.Add(m_Ua);
            response.Add(m_Wa);
            response.Add(m_bUa);
            response.Add(m_bWa);
            response.Add(m_V);

            return response;
        }

        public void Save(Stream stream)
        {
            m_Ua.Save(stream);
            m_Wa.Save(stream);
            m_bUa.Save(stream);
            m_bWa.Save(stream);
            m_V.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_Ua.Load(stream);
            m_Wa.Load(stream);
            m_bUa.Load(stream);
            m_bWa.Load(stream);
            m_V.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            AttentionUnit a = new AttentionUnit(m_name, m_hiddenDim, m_contextDim, deviceId);          
            return a;
        }
    }
}



