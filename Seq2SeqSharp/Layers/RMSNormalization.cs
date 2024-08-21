// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp
{
    [Serializable]
    internal class RMSNormalization : INormalization
    {
        private readonly IWeightTensor m_alpha;
        private readonly float m_epsilon;

        public RMSNormalization(string name, int dim, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, float epsilon = 1e-06f, DType elementType = DType.Float32)
        {
            m_alpha = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(m_alpha)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_epsilon = epsilon;
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            var result = g.RMSNorm(input, m_alpha, m_epsilon);
            return result;
        }
      
        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_alpha,
            };

            return response;
        }

        public void Save(IModel stream)
        {
            m_alpha.Save(stream);
        }


        public void Load(IModel stream)
        {
            m_alpha.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            throw new NotImplementedException();
        }

        public int GetDeviceId()
        {
            throw new NotImplementedException();
        }
    }
}
