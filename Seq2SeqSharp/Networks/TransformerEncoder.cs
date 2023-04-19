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
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp
{
    internal class TransformerEncoder : IEncoder
    {
        private readonly List<MultiHeadAttention> m_encoders = new List<MultiHeadAttention>();
        private readonly List<IFeedForwardLayer> m_feedForwards = new List<IFeedForwardLayer>();

        private readonly int m_inputDim;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly int m_multiHeadNum;
        private readonly int m_hiddenDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly float m_learningRateFactor;
        private readonly LayerNormalization layerNorm;
        private readonly ActivateFuncEnums m_activateFunc;
        private readonly int m_expertNum;
        private readonly int m_expertsPerTokenFactor;
        private readonly DType m_elementType;

        public TransformerEncoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.Relu, int expertNum = 1, int expertsPerTokenFactor = 1, DType elementType = DType.Float32)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}', ElementType = '{elementType}'");

            m_name = name;
            m_multiHeadNum = multiHeadNum;
            m_hiddenDim = hiddenDim;
            m_inputDim = inputDim;
            m_depth = depth;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;
            m_learningRateFactor = learningRateFactor;
            m_activateFunc = activateFunc;
            m_expertNum = expertNum;
            m_expertsPerTokenFactor = expertsPerTokenFactor;
            m_elementType = elementType;

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerEncoder.");
            }

            m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType));
            for (int i = 1; i < depth; i++)
            {
                m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType));
            }

            for (int i = 0; i < depth; i++)
            {
                if (m_expertNum > 1 && i % 2 == 0)
                {
                    m_feedForwards.Add(new MoEFeedForward($"{name}.MoEFFN_{i}", m_expertNum, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, expertsPerTokenFactor: expertsPerTokenFactor));
                }
                else
                {
                    m_feedForwards.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, elementType: elementType));
                }
            }

            layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor);

        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
        }

        /// <summary>
        /// Transformer encoder
        /// </summary>
        /// <param name="rawInputs"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightTensor Encode(IWeightTensor inputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            using (IComputeGraph subg = g.CreateSubGraph($"{m_name}_Encoder"))
            {
                IWeightTensor maskTensor = null;
                if (srcSelfMask != null)
                {
                    int seqLen = inputs.Rows / batchSize;
                    using var keyMaskView = subg.View(srcSelfMask, dims: new long[] { batchSize, 1, seqLen, seqLen });
                    maskTensor = subg.Expand(keyMaskView, dims: new long[] { batchSize, m_multiHeadNum, seqLen, seqLen });
                }

                IWeightTensor attnProbs = null;
                for (int k = 0; k < m_encoders.Count; k++)
                {
                    var inputs2 = m_encoders[k].Perform(inputs, maskTensor, batchSize, subg);
                    inputs.ReleaseWeight();

                    inputs = m_feedForwards[k].Process(inputs2, batchSize, subg);
                    inputs2.ReleaseWeight();
                }

                inputs = layerNorm.Norm(inputs, subg);

                inputs.UnbindFromComputeGraph();
                if (attnProbs != null)
                {
                    attnProbs.UnbindFromComputeGraph();
                }

                if (maskTensor != null)
                {
                    maskTensor.Dispose();
                }
            }

            return inputs;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerEncoder(m_name, m_multiHeadNum, m_hiddenDim, m_inputDim, m_depth, m_dropoutRatio, deviceId, m_isTrainable, learningRateFactor: m_learningRateFactor, activateFunc: m_activateFunc, expertNum: m_expertNum, expertsPerTokenFactor: m_expertsPerTokenFactor, elementType: m_elementType);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (MultiHeadAttention item in m_encoders)
            {
                response.AddRange(item.GetParams());
            }

            foreach (var item in m_feedForwards)
            {
                response.AddRange(item.GetParams());
            }

            response.AddRange(layerNorm.GetParams());

            return response;
        }

        public void Save(IModel stream)
        {
            foreach (MultiHeadAttention item in m_encoders)
            {
                item.Save(stream);
            }

            foreach (var item in m_feedForwards)
            {
                item.Save(stream);
            }

            layerNorm.Save(stream);
        }

        public void Load(IModel stream)
        {
            foreach (MultiHeadAttention item in m_encoders)
            {
                item.Load(stream);
            }

            foreach (var item in m_feedForwards)
            {
                item.Load(stream);
            }

            layerNorm.Load(stream);
        }
    }
}
