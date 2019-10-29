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
        string m_name;

        public LayerNormalization(string name, int dim, int deviceId)
        {
            m_name = name;
            m_alpha = new WeightTensor(new long[2] { 1, dim }, 1, deviceId, name: $"{name}.{nameof(m_alpha)}", isTrainable: true);
            m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_beta)}", isTrainable: true);
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            var innerGraph = g.CreateSubGraph(m_name);
            return innerGraph.LayerNorm(input, m_alpha, m_beta);
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
            var innerGraph = g.CreateSubGraph(m_name);
            return innerGraph.AddLayerNorm(input1, input2, m_alpha, m_beta);
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
