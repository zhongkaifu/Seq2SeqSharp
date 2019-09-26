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
        public IWeightMatrix uhs;
        public IWeightMatrix inputs;

    }

    [Serializable]
    public class AttentionUnit
    {

        public IWeightMatrix V { get; set; }
        public IWeightMatrix Ua { get; set; }
        public IWeightMatrix bUa { get; set; }
        public IWeightMatrix Wa { get; set; }
        public IWeightMatrix bWa { get; set; }

        int m_batchSize;

        public AttentionUnit(int batchSize, int size, int context, ArchTypeEnums archType, int deviceId)
        {
            m_batchSize = batchSize;

            this.Ua = new WeightTensor(context, size, deviceId);
            this.Wa = new WeightTensor(size, size, deviceId);
            this.bUa = new WeightTensor(1, size, 0, deviceId);
            this.bWa = new WeightTensor(1, size, 0, deviceId);
            this.V = new WeightTensor(size, 1, deviceId);
        }



        public AttentionPreProcessResult PreProcess(IWeightMatrix inputs, IComputeGraph g)
        {
            AttentionPreProcessResult r = new AttentionPreProcessResult();

            IWeightMatrix bUas = g.RepeatRows(bUa, inputs.Rows);
            r.uhs = g.MulAdd(inputs, Ua, bUas);
            r.inputs = g.PermuteBatch(inputs, m_batchSize);

            return r;
        }

      

        public IWeightMatrix Perform(IWeightMatrix state, AttentionPreProcessResult attenPreProcessResult, IComputeGraph g)
        {
            var bWas = g.RepeatRows(bWa, state.Rows);
            var wc = g.MulAdd(state, Wa, bWas);
            var wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / m_batchSize);
            var ggs = g.AddTanh(attenPreProcessResult.uhs, wcs);
            var atten = g.Mul(ggs, V);

            var atten2 = g.PermuteBatch(atten, m_batchSize);
            var attenT = g.Transpose2(atten2);
            var attenT2 = g.View(attenT, m_batchSize, attenPreProcessResult.inputs.Rows / m_batchSize);

            var attenSoftmax1 = g.Softmax(attenT2);

            var attenSoftmax = g.View(attenSoftmax1, m_batchSize, attenSoftmax1.Rows / m_batchSize, attenSoftmax1.Columns);
            var inputs2 = g.View(attenPreProcessResult.inputs, m_batchSize, attenPreProcessResult.inputs.Rows / m_batchSize, attenPreProcessResult.inputs.Columns);

            IWeightMatrix contexts = g.MulBatch(attenSoftmax, inputs2, m_batchSize);


            return contexts;
        }

      

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            response.Add(Ua);
            response.Add(Wa);
            response.Add(bUa);
            response.Add(bWa);
            response.Add(V);

            return response;
        }

        public void Save(Stream stream)
        {
            Ua.Save(stream);
            Wa.Save(stream);
            bUa.Save(stream);
            bWa.Save(stream);
            V.Save(stream);
        }


        public void Load(Stream stream)
        {
            Ua.Load(stream);
            Wa.Load(stream);
            bUa.Load(stream);
            bWa.Load(stream);
            V.Load(stream);
        }
    }
}



