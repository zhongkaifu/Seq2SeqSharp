using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    [Serializable]
    internal class LayerNormalization
    {
        private readonly IWeightTensor m_alpha;
        private readonly IWeightTensor m_beta;
        private readonly string m_name;

        public LayerNormalization(string name, int dim, int deviceId, bool isTrainable)
        {
            m_name = name;
            m_alpha = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(m_alpha)}", isTrainable: isTrainable);
            m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_beta)}", isTrainable: isTrainable);
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            return g.LayerNorm(input, m_alpha, m_beta, 1e-06f);
        }

        /// <summary>
        /// LayerNorm (input1 + input2)
        /// </summary>
        /// <param name="input1"></param>
        /// <param name="input2"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightTensor AddNorm(IWeightTensor input1, IWeightTensor input2, IComputeGraph g)
        {
            return g.AddLayerNorm(input1, input2, m_alpha, m_beta);
        }

        public virtual List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_alpha,
                m_beta
            };

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
