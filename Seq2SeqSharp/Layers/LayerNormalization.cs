// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using ManagedCuda.VectorTypes;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Xml.Linq;
using TensorSharp;

namespace Seq2SeqSharp
{
    [Serializable]
    internal class LayerNormalization : INormalization
    {
        private readonly IWeightTensor m_alpha;
        private readonly IWeightTensor m_beta;
        private readonly float m_epsilon;

        private readonly string m_name;
        private readonly int m_dim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly float m_learningRateFactor;
        private readonly DType m_elementType;


        public LayerNormalization(string name, int dim, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, float epsilon = 1e-06f, DType elementType = DType.Float32)
        {
            m_name = name;
            m_dim = dim;
            m_deviceId = deviceId;
            m_isTrainable= isTrainable;
            m_learningRateFactor= learningRateFactor;
            m_elementType= elementType;

            m_alpha = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(m_alpha)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_beta)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_epsilon = epsilon;
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            var result = g.LayerNorm(input, m_alpha, m_beta, m_epsilon);
            return result;
        }

        ///// <summary>
        ///// LayerNorm (input1 + input2)
        ///// </summary>
        ///// <param name="input1"></param>
        ///// <param name="input2"></param>
        ///// <param name="g"></param>
        ///// <returns></returns>
        //public IWeightTensor AddNorm(IWeightTensor input1, IWeightTensor input2, IComputeGraph g)
        //{
        //    return g.AddLayerNorm(input1, input2, m_alpha, m_beta);
        //}

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_alpha,
                m_beta
            };

            return response;
        }

        public void Save(IModel stream)
        {
            m_alpha.Save(stream);
            m_beta.Save(stream);
        }


        public void Load(IModel stream)
        {
            m_alpha.Load(stream);
            m_beta.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new LayerNormalization(m_name, m_dim, deviceId, m_isTrainable, m_learningRateFactor, m_epsilon, m_elementType);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }
    }
}
