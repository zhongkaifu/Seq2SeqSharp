using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    [Serializable]
    class LayerNormalization
    {
        IWeightTensor m_alpha;
        IWeightTensor m_beta;

        public LayerNormalization(int dim, int deviceId)
        {
            m_alpha = new WeightTensor(1, dim, 1, deviceId);
            m_beta = new WeightTensor(1, dim, 0, deviceId);
        }

        public IWeightTensor Process(IWeightTensor input, IComputeGraph innerGraph)
        {
            var alphas = innerGraph.RepeatRows(m_alpha, input.Rows);
            var betas = innerGraph.RepeatRows(m_beta, input.Rows);

            return innerGraph.LayerNorm(input, alphas, betas);
        }

        public virtual List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();
            response.Add(m_alpha);
            response.Add(m_beta);

            return response;
        }

        public void Save(Stream stream)
        {
            m_alpha.Save(stream);
            m_beta.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_alpha.Load(stream);
            m_beta.Load(stream);
        }
    }
}
