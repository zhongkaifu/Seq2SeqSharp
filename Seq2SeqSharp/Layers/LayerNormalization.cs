// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    [Serializable]
    internal class LayerNormalization
    {
        private readonly IWeightTensor m_alpha;
        private readonly IWeightTensor m_beta;
        private readonly float m_epsilon;

        public LayerNormalization(string name, int dim, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, float epsilon = 1e-06f)
        {
            m_alpha = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(m_alpha)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_beta)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_epsilon = epsilon;
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            return g.LayerNorm(input, m_alpha, m_beta, m_epsilon);
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
    }
}
