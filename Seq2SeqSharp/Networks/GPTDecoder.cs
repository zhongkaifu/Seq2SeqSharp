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
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp
{
    public class GPTDecoder : IDecoder
    {
        private readonly List<IAttentionLayer> m_selfAttns = new List<IAttentionLayer>();
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
        private readonly INormalization layerNorm;
        private readonly ActivateFuncEnums m_activateFunc;
        private readonly int m_expertNum;
        private readonly int m_expertsPerTokenFactor;
        private readonly DType m_elementType;
        private readonly PositionEmbeddingEnums m_peType;
        private readonly NormEnums m_normType;
        private readonly AttentionTypeEnums m_attentionType;
        private readonly MultiHeadAttentionTypeEnums m_multiHeadAttentionType;
        private readonly int m_KVGroupNum;

        public AttentionTypeEnums AttentionType => m_attentionType;
        

        public GPTDecoder(string name, int multiHeadNum, int hiddenDim, int intermediateDim, int inputDim, int depth, float dropoutRatio, int deviceId, 
            bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.ReLU, int expertNum = 1, 
            int expertsPerTokenFactor = 1, DType elementType = DType.Float32, PositionEmbeddingEnums peType = PositionEmbeddingEnums.APE, NormEnums normType = NormEnums.LayerNorm, 
            AttentionTypeEnums attentionType = AttentionTypeEnums.Classic, MultiHeadAttentionTypeEnums multiHeadAttentionType = MultiHeadAttentionTypeEnums.MHA, int KVGroupNum = 0)
        {
            Logger.WriteLine(Logger.Level.debug, $"Creating transformer decoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', IntermediateDim = '{intermediateDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}', ElementType = '{elementType}', Positional Embedding = '{peType}'， Norm = '{normType}' AttentionType = '{attentionType}' Multi-Head Attention Type = '{multiHeadAttentionType}'");

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
            m_elementType= elementType;
            m_peType = peType;
            m_normType = normType;
            m_attentionType = attentionType;
            m_multiHeadAttentionType = multiHeadAttentionType;
            m_KVGroupNum = KVGroupNum;

            if (multiHeadAttentionType == MultiHeadAttentionTypeEnums.GQA)
            {
                Logger.WriteLine(Logger.Level.debug, $"The number of KV Group = '{m_KVGroupNum}'");
            }

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in GPTDecoder.");
            }

            if (m_multiHeadAttentionType == MultiHeadAttentionTypeEnums.GQA)
            {
                m_selfAttns.Add(new GroupQueryAttention(name: $"{name}.SelfAttn_0", num_heads: multiHeadNum, num_kv_groups: m_KVGroupNum, d_out: hiddenDim, d_in: inputDim, dropoutRatio: m_dropoutRatio, deviceId: deviceId,
    isTrainable: isTrainable, learningRateFactor: learningRateFactor, elementType: elementType, peType: peType, normType: normType));
            }
            else
            {
                m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId,
                    isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType, peType: peType, normType: normType, attentionType: m_attentionType));
            }


            for (int i = 1; i < depth; i++)
            {
                if (multiHeadAttentionType == MultiHeadAttentionTypeEnums.GQA)
                {
                    m_selfAttns.Add(new GroupQueryAttention(name: $"{name}.SelfAttn_{i}", num_heads: multiHeadNum, num_kv_groups: m_KVGroupNum, d_out: hiddenDim, d_in: hiddenDim, dropoutRatio: m_dropoutRatio, deviceId: deviceId,
isTrainable: isTrainable, learningRateFactor: learningRateFactor, elementType: elementType, peType: peType, normType: normType));
                }
                else
                {
                    m_selfAttns.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId,
                        isTrainable: isTrainable, sharedQKV: true, learningRateFactor: learningRateFactor, elementType: elementType, peType: peType, normType: normType, attentionType: m_attentionType));
                }
            }

            for (int i = 0; i < depth; i++)
            {
                if (m_expertNum > 1 && i > 1)
                {
                    m_feedForwards.Add(new MoEFeedForward($"{name}.MoEFFN_{i}", m_expertNum, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, elementType: elementType, expertsPerTokenFactor: expertsPerTokenFactor));
                }
                else
                {
                    m_feedForwards.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, intermediateDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, activateFunc: activateFunc, elementType: elementType, normType: normType));
                }
            }

            if (normType == NormEnums.LayerNorm)
            {
                layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
            else
            {
                layerNorm = new RMSNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
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

        public (IWeightTensor, IWeightTensor) Decode(IWeightTensor tgtInputs, IWeightTensor tgtSelfMask, int batchSize, IComputeGraph g, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            IWeightTensor attnProbs = null;
            IWeightTensor tgtInputsFinal = null;
            using (IComputeGraph subg = g.CreateSubGraph($"{m_name}_GPTDecoder"))
            {
                int seqLenQ = tgtInputs.Rows / batchSize;
                IWeightTensor selfMaskTensor = null;
                if (tgtSelfMask != null)
                {
                    selfMaskTensor = subg.Expand(tgtSelfMask, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenQ });
                }

                for (int k = 0; k < m_selfAttns.Count; k++)
                {
                    var tgtInputs2 = m_selfAttns[k].Perform(tgtInputs, selfMaskTensor, batchSize, subg, cachedTensors: cachedTensors);
                    tgtInputs.ReleaseWeight();

                    tgtInputs = m_feedForwards[k].Process(tgtInputs2, batchSize, subg, cachedTensors: cachedTensors);
                    tgtInputs2.ReleaseWeight();
                }

                tgtInputsFinal = layerNorm.Norm(tgtInputs, subg);
                tgtInputs.ReleaseWeight();

                tgtInputsFinal.UnbindFromComputeGraph();
                if (attnProbs != null)
                {
                    attnProbs.UnbindFromComputeGraph();
                }

                if (selfMaskTensor != null)
                {
                    selfMaskTensor.Dispose();
                }
            }

            return (tgtInputsFinal, attnProbs);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new GPTDecoder(m_name, m_multiHeadNum, m_hiddenDim, m_intermediateDim, m_inputDim, m_depth, m_dropoutRatio, deviceId, m_isTrainable, learningRateFactor: m_learningRateFactor, activateFunc: m_activateFunc, expertNum: m_expertNum, 
                expertsPerTokenFactor: m_expertsPerTokenFactor, elementType: m_elementType, peType: m_peType, normType: m_normType, attentionType: m_attentionType, multiHeadAttentionType: m_multiHeadAttentionType, KVGroupNum: m_KVGroupNum);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (IAttentionLayer item in m_selfAttns)
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
            foreach (IAttentionLayer item in m_selfAttns)
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
            foreach (IAttentionLayer item in m_selfAttns)
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
