using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    class FeedForwardLayer
    {
        private IWeightMatrix m_Whd;
        private IWeightMatrix m_Bd;

        public FeedForwardLayer(int inputDim, int outputDim, ArchTypeEnums archType, int deviceId)
        {
            m_Whd = new WeightTensor(inputDim, outputDim, deviceId);
            m_Bd = new WeightTensor(1, outputDim, 0, deviceId);
        }

        public IWeightMatrix Process(IWeightMatrix inputT, IComputeGraph g)
        {
            var bds = g.RepeatRows(m_Bd, inputT.Rows);
            var r = g.MulAdd(inputT, m_Whd, bds);

            return r;
        }

        public virtual List<IWeightMatrix> GetParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
            response.Add(m_Whd);
            response.Add(m_Bd);

            return response;
        }

        public void Save(Stream stream)
        {
            m_Whd.Save(stream);
            m_Bd.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_Whd.Load(stream);
            m_Bd.Load(stream);
        }
    }
}
