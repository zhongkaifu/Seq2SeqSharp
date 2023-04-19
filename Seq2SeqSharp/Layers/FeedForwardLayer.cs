// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp
{
    internal class FeedForwardLayer : IFeedForwardLayer
    {
        private readonly IWeightTensor m_Whd;
        private readonly IWeightTensor m_Bd;
        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly int m_inputDim;
        private readonly int m_outputDim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly DType m_elementType;

        public FeedForwardLayer(string name, int inputDim, int outputDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, DType elementType = DType.Float32)
        {
            Logger.WriteLine($"Create feed forward layer '{name}' InputDim = '{inputDim}', OutputDim = '{outputDim}', DropoutRatio = '{dropoutRatio}', DeviceId = '{deviceId}'");

            m_name = name;
            m_inputDim = inputDim;
            m_outputDim = outputDim;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;
            m_elementType = elementType;

            m_Whd = new WeightTensor(new long[2] { inputDim, outputDim }, deviceId, name: $"{name}.{nameof(m_Whd)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_Bd = new WeightTensor(new long[2] { 1, outputDim }, 0, deviceId, name: $"{name}.{nameof(m_Bd)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public void ClearStatus()
        {

        }

        public IWeightTensor Process(IWeightTensor inputT, int batchSize, IComputeGraph g, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            IWeightTensor res = null;

            if (inputT.ElementType == DType.Float16)
            {

                var whd = g.Float2Half(m_Whd);
                var bd = g.Float2Half(m_Bd);

                res = g.Affine(inputT, whd, bd, 1.0f);
            }
            else
            {
                res = g.Affine(inputT, m_Whd, m_Bd, 1.0f);
            }


            return g.Dropout(res, batchSize, m_dropoutRatio, inPlace: true);
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_Whd,
                m_Bd
            };

            return response;
        }

        public void Save(IModel stream)
        {
            m_Whd.Save(stream);
            m_Bd.Save(stream);
        }


        public void Load(IModel stream)
        {
            m_Whd.Load(stream);
            m_Bd.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new FeedForwardLayer(m_name, m_inputDim, m_outputDim, m_dropoutRatio, deviceId, m_isTrainable, elementType: m_elementType);
        }
    }
}
