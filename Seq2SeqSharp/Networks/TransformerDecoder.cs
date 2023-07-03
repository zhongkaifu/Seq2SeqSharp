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
    public class TransformerDecoder : IDecoder
    {
        private readonly List<MultiHeadAttention> m_selfAttns = new List<MultiHeadAttention>();
        private readonly List<MultiHeadAttention> m_encAttns = new List<MultiHeadAttention>();
        private readonly List<IFeedForwardLayer> m_feedForwards = new List<IFeedForwardLayer>();

        private readonly int m_inputDim;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly int m_multiHeadNum;
        private readonly int m_hiddenDim;
        private readonly int m_intermediateDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly float m_learningRateFactor;
        private readonly LayerNormalization layerNorm;
        private readonly ActivateFuncEnums m_activateFunc;
        private readonly int m_expertNum;
        private readonly int m_expertsPerTokenFactor;
        private readonly DType m_elementType;

        public TransformerDecoder(string name, int multiHeadNum, int hiddenDim, int intermediateDim, int inputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.Relu, int expertNum = 1, int expertsPerTokenFactor = 1, DType elementType = DType.Float32)
        {
            Logger.WriteLine($"Creating transformer decoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', IntermediateDim = '{intermediateDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}', ElementType = '{elementType}'");

            m_name = name;
            m_multiHeadNum = multiHeadNum;
            m_hiddenDim = hiddenDim;
            m_intermediateDim = intermediateDim;
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
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerDecoder.");
            }

            m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType));
            for (int i = 1; i < depth; i++)
            {
                m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType));
            }

            m_encAttns.Add(new MultiHeadAttention($"{name}.EncAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, learningRateFactor: learningRateFactor, elementType: elementType));
            for (int i = 1; i < depth; i++)
            {
                m_encAttns.Add(new MultiHeadAttention($"{name}.EncAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, learningRateFactor: learningRateFactor, elementType: elementType));
            }

            for (int i = 0; i < depth; i++)
            {
                if (m_expertNum > 1 && i % 2 == 0)
                {
                    m_feedForwards.Add(new MoEFeedForward($"{name}.MoEFFN_{i}", m_expertNum, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, expertsPerTokenFactor: expertsPerTokenFactor));
                }
                else
                {
                    m_feedForwards.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, intermediateDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, elementType: elementType));
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
        /// 

        public (IWeightTensor, IWeightTensor) Decode(IWeightTensor tgtInputs, IWeightTensor encOutputBatchFirst, IWeightTensor tgtSelfMask, IWeightTensor srcTgtMask, int batchSize, IComputeGraph g, bool outputAttnWeights = false, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            IWeightTensor attnProbs = null;
            using (IComputeGraph subg = g.CreateSubGraph($"{m_name}_Decoder"))
            {
                int seqLenQ = tgtInputs.Rows / batchSize;

                // SeqLenK must be euqal to SeqLenV
                int seqLenK = encOutputBatchFirst.Rows / batchSize;

                IWeightTensor selfMaskTensor = null;
                if (tgtSelfMask != null)
                {
                    selfMaskTensor = subg.Expand(tgtSelfMask, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenQ });
                }

                IWeightTensor crossMaskTensor = null;
                if (srcTgtMask != null)
                {
                    crossMaskTensor = subg.Expand(srcTgtMask, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenK });
                }

                for (int k = 0; k < m_selfAttns.Count; k++)
                {
                    var tgtInputs2 = m_selfAttns[k].Perform(tgtInputs, selfMaskTensor, batchSize, subg, cachedTensors: cachedTensors);
                    tgtInputs.ReleaseWeight();

                    (var tgtInputs3, attnProbs) = m_encAttns[k].Perform(tgtInputs2, encOutputBatchFirst, encOutputBatchFirst, crossMaskTensor, batchSize, subg, outputAttenWeights: (outputAttnWeights && k == m_selfAttns.Count - 1), cachedTensors: cachedTensors);
                    tgtInputs2.ReleaseWeight();

                    tgtInputs = m_feedForwards[k].Process(tgtInputs3, batchSize, subg, cachedTensors: cachedTensors);
                    tgtInputs3.ReleaseWeight();
                }

                tgtInputs = layerNorm.Norm(tgtInputs, subg);

                tgtInputs.UnbindFromComputeGraph();
                if (attnProbs != null)
                {
                    attnProbs.UnbindFromComputeGraph();
                }

                if (selfMaskTensor != null)
                {
                    selfMaskTensor.Dispose();
                }

                if (crossMaskTensor != null)
                {
                    crossMaskTensor.Dispose();
                }
            }
            
            return (tgtInputs, attnProbs);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerDecoder(m_name, m_multiHeadNum, m_hiddenDim, m_intermediateDim, m_inputDim, m_depth, m_dropoutRatio, deviceId, m_isTrainable, learningRateFactor: m_learningRateFactor, activateFunc: m_activateFunc, expertNum: m_expertNum, expertsPerTokenFactor: m_expertsPerTokenFactor, elementType: m_elementType);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (MultiHeadAttention item in m_selfAttns)
            {
                response.AddRange(item.GetParams());
            }

            foreach (MultiHeadAttention item in m_encAttns)
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
            foreach (MultiHeadAttention item in m_selfAttns)
            {
                item.Save(stream);
            }

            foreach (MultiHeadAttention item in m_encAttns)
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
            foreach (MultiHeadAttention item in m_selfAttns)
            {
                item.Load(stream);
            }

            foreach (MultiHeadAttention item in m_encAttns)
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
