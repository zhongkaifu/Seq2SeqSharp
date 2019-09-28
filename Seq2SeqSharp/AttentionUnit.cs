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

        public AttentionUnit(int batchSize, int size, int context, int deviceId)
        {
            m_batchSize = batchSize;

            m_Ua = new WeightTensor(context, size, deviceId, true);
            m_Wa = new WeightTensor(size, size, deviceId, true);
            m_bUa = new WeightTensor(1, size, 0, deviceId);
            m_bWa = new WeightTensor(1, size, 0, deviceId);
            m_V = new WeightTensor(size, 1, deviceId, true);
        }

        public AttentionPreProcessResult PreProcess(IWeightTensor inputs, IComputeGraph g)
        {
            AttentionPreProcessResult r = new AttentionPreProcessResult();

            IWeightTensor bUas = g.RepeatRows(m_bUa, inputs.Rows);
            r.uhs = g.MulAdd(inputs, m_Ua, bUas);
            r.inputs = g.PermuteBatch(inputs, m_batchSize);

            return r;
        }

        public IWeightTensor Perform(IWeightTensor state, AttentionPreProcessResult attenPreProcessResult, IComputeGraph g)
        {
            var bWas = g.RepeatRows(m_bWa, state.Rows);
            var wc = g.MulAdd(state, m_Wa, bWas);
            var wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / m_batchSize);
            var ggs = g.AddTanh(attenPreProcessResult.uhs, wcs);
            var atten = g.Mul(ggs, m_V);

            var atten2 = g.PermuteBatch(atten, m_batchSize);
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



