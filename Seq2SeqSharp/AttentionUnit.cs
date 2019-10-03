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
    public class AttentionUnit
    {
        IWeightTensor m_V;
        IWeightTensor m_Ua;
        IWeightTensor m_bUa;
        IWeightTensor m_Wa;
        IWeightTensor m_bWa;

        int m_batchSize;
        string m_name;

        public AttentionUnit(string name, int batchSize, int size, int context, int deviceId)
        {
            m_name = name;
            m_batchSize = batchSize;

            m_Ua = new WeightTensor(new long[2] { context, size }, deviceId, normal: true, name: $"{name}.{nameof(m_Ua)}", isTrainable: true);
            m_Wa = new WeightTensor(new long[2] { size, size }, deviceId, normal:true, name: $"{name}.{nameof(m_Wa)}", isTrainable: true);
            m_bUa = new WeightTensor(new long[2] { 1, size }, 0, deviceId, name: $"{name}.{nameof(m_bUa)}", isTrainable: true);
            m_bWa = new WeightTensor(new long[2] { 1, size }, 0, deviceId, name: $"{name}.{nameof(m_bWa)}", isTrainable: true);
            m_V = new WeightTensor(new long[2] { size, 1 }, deviceId, normal:true, name: $"{name}.{nameof(m_V)}", isTrainable: true);
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor inputs, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name + "_PreProcess");
            AttentionPreProcessResult r = new AttentionPreProcessResult();

            r.uhs = g.Affine(inputs, m_Ua, m_bUa);
            r.inputs = g.TransposeBatch(inputs, m_batchSize);

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attenPreProcessResult, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name);

            var wc = g.Affine(state, m_Wa, m_bWa);
            var wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / m_batchSize);
            var ggs = g.AddTanh(attenPreProcessResult.uhs, wcs);
            var atten = g.Mul(ggs, m_V);

            var atten2 = g.TransposeBatch(atten, m_batchSize);
            var attenT = g.Transpose(atten2);
            var attenT2 = g.View(attenT, m_batchSize, attenPreProcessResult.inputs.Rows / m_batchSize);

            var attenSoftmax1 = g.Softmax(attenT2);

            var attenSoftmax = g.View(attenSoftmax1, m_batchSize, attenSoftmax1.Rows / m_batchSize, attenSoftmax1.Columns);
            var inputs2 = g.View(attenPreProcessResult.inputs, m_batchSize, attenPreProcessResult.inputs.Rows / m_batchSize, attenPreProcessResult.inputs.Columns);

            IWeightTensor contexts = g.MulBatch(attenSoftmax, inputs2, m_batchSize);


            return contexts;
        }

      

        public virtual List<IWeightTensor> getParams()
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
    }
}



