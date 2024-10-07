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
    internal class MultiHeadAttention : IAttentionLayer
    {
        private readonly IWeightTensor W0;
        private readonly IWeightTensor b0;

        private readonly IWeightTensor Q;
        private readonly IWeightTensor K;
        private readonly IWeightTensor V;

        private readonly IWeightTensor Qb;
        private readonly IWeightTensor Kb;
        private readonly IWeightTensor Vb;

        private readonly IWeightTensor QKV;
        private readonly IWeightTensor QKVb;


        private readonly INormalization layerNormQ;

        private readonly int m_hiddenDim;
        private readonly int m_d;
        private readonly int m_multiHeadNum;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        private readonly bool m_sharedQKV;

        private readonly PositionEmbeddingEnums m_PEType;
        private readonly AttentionTypeEnums m_attentionType;

        public MultiHeadAttention(string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId, bool isTrainable, 
            bool sharedQKV = false, float learningRateFactor = 1.0f, DType elementType = DType.Float32, PositionEmbeddingEnums peType = PositionEmbeddingEnums.APE, NormEnums normType = NormEnums.LayerNorm, AttentionTypeEnums attentionType = AttentionTypeEnums.Classic)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_multiHeadNum = multiHeadNum;

            if (m_hiddenDim % m_multiHeadNum != 0)
            {
                throw new ArgumentException("The hidden dim must be divisible by multi-head size.");
            }

            m_d = m_hiddenDim / m_multiHeadNum;
            m_dropoutRatio = dropoutRatio;
            m_sharedQKV = sharedQKV;
            m_PEType = peType;
            m_attentionType = attentionType;

            Logger.WriteLine(Logger.Level.debug, $"Creating multi-head attention layer. Name = '{name}', HiddenDim = '{hiddenDim}', multi-head dim = '{multiHeadNum}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', PE = '{peType}', Norm = '{normType}' AttentionType = '{attentionType}'");

            W0 = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.{nameof(W0)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
            b0 = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(b0)}", isTrainable: isTrainable, dtype: elementType);

            if (m_sharedQKV == false)
            {
                Q = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
                Qb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

                K = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(K)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
                Kb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Kb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

                V = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(V)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
                Vb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Vb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            }
            else
            {
                QKV = new WeightTensor(new long[2] { inputDim, hiddenDim * 3 }, deviceId, name: $"{name}.{nameof(QKV)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
                QKVb = new WeightTensor(new long[2] { 1, hiddenDim * 3 }, 0, deviceId, name: $"{name}.{nameof(QKVb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            }

            if (normType == NormEnums.LayerNorm)
            {
                layerNormQ = new LayerNormalization($"{name}.{nameof(layerNormQ)}", m_hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
            else
            {
                layerNormQ = new RMSNormalization($"{name}.{nameof(layerNormQ)}", m_hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
        }

        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            if (m_attentionType == AttentionTypeEnums.Classic)
            {
                return PerformClassic(inputQ, keyMask, batchSize, graph, cachedTensors);
            }
            else
            {
                return PerformFlashAttentionWithCausal(inputQ, batchSize, graph, cachedTensors);
            }
        }


        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        private IWeightTensor PerformClassic(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            string keyName = $"{m_name}_MultiHeadAttention_1";
            using IComputeGraph g = graph.CreateSubGraph(keyName);
            int seqLenQ = inputQ.Rows / batchSize;
            IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

            //Input projections
            var weightedQKV = g.View(g.Affine(inputQNorm, QKV, QKVb), dims: new long[] { batchSize, seqLenQ, 3, m_multiHeadNum, m_d });
            var allQ = g.Select(weightedQKV, 2, 0);
            var allK = g.Select(weightedQKV, 2, 1);
            var allV = g.Select(weightedQKV, 2, 2);

            //Multi-head attentions
            IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
            if (m_PEType == PositionEmbeddingEnums.RoPE)
            {
                Qs = g.RoPE(Qs, seqLenQ);
            }

            int newTokensIdx = seqLenQ;
            IWeightTensor m_cacheQs = null;
            string QKeyName = keyName + "_" + nameof(inputQ);
            if (cachedTensors != null)
            {
                if (cachedTensors.ContainsKey(QKeyName) == true)
                {
                    m_cacheQs = cachedTensors[QKeyName];
                    newTokensIdx = seqLenQ - (int)m_cacheQs.Sizes[0];
                }
                else
                {
                    cachedTensors.Add(QKeyName, null);
                }

                // Optimize runtime for test that only processing new tokens.
                // For older tokens, just use cached output
                Qs = g.Peek(Qs, 1, seqLenQ - newTokensIdx, newTokensIdx); // Shape: [batchSize * m_multiHeadNum, relPosSize, m_d]
            }

            IWeightTensor Ks = null;
            if (m_PEType == PositionEmbeddingEnums.RoPE)
            {
                Ks = g.View(g.AsContiguous(g.Transpose(allK, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                Ks = g.RoPE(Ks, seqLenQ);
                Ks = g.View(g.AsContiguous(g.Transpose(Ks, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenQ });
            }
            else
            {
                Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenQ });
            }

            IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });

            // Scaled masked softmax
            float scale = 1.0f / (float)(Math.Sqrt(m_d));

            // Convert tensors to Float32 type if they are not that type.
            bool useF16 = (Qs.ElementType == DType.Float16);
            if (useF16)
            {
                Qs = g.Half2Float(Qs);
                Ks = g.Half2Float(Ks);
            }
            var attn = g.MulBatch(Qs, Ks, scale); // Shape: [batchSize * m_multiHeadNum, relPosSize, seqLenQ]

            // Add mask
            attn = g.View(attn, dims: new long[] { batchSize, m_multiHeadNum, newTokensIdx, seqLenQ });
            if (keyMask != null)
            {
                if (cachedTensors != null)
                {
                    keyMask = g.Peek(keyMask, 2, seqLenQ - newTokensIdx, newTokensIdx);
                }
                if (useF16)
                {
                    keyMask = g.Half2Float(keyMask);
                }
                attn = g.Add(attn, keyMask, inPlace: true);
            }
            attn = g.Softmax(attn, inPlace: true);

            // Convert it back to Float16 for the following parts
            if (useF16)
            {
                attn = g.Float2Half(attn);
            }

            attn = g.View(attn, dims: new long[] { batchSize * m_multiHeadNum, newTokensIdx, seqLenQ });
            IWeightTensor o = g.View(g.MulBatch(attn, Vs), dims: new long[] { batchSize, m_multiHeadNum, newTokensIdx, m_d });
            IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * newTokensIdx, m_multiHeadNum * m_d });

            // Output projection
            IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), m_dropoutRatio, inPlace: true); // Shape: [batchSize * relPosSize, m_multiHeadNum * m_d]

            if (cachedTensors != null)
            {
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize, newTokensIdx, m_multiHeadNum * m_d });
                finalAttResults = g.Transpose(finalAttResults, 0, 1); // Shape: [relPosSize, batchSize, m_multiHeadNum * m_d]

                if (m_cacheQs == null)
                {
                    m_cacheQs = finalAttResults;
                }
                else
                {
                    m_cacheQs = g.Concate(0, m_cacheQs, finalAttResults);
                }
                m_cacheQs.UnbindFromComputeGraph();

                cachedTensors[QKeyName] = m_cacheQs;

                finalAttResults = g.AsContiguous(g.Transpose(m_cacheQs, 0, 1)); // Shape: [batchSize, seqLenQ, m_multiHeadNum * m_d]
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });
            }

            // For runtime, we don't call it by inplace, since it will change the content of cached weights
            IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: true);

            return result;
        }




        private IWeightTensor PerformFlashAttentionWithCausal(IWeightTensor inputQ, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            string keyName = $"{m_name}_MultiHeadAttention_1";
            using IComputeGraph g = graph.CreateSubGraph(keyName);
            int seqLenQ = inputQ.Rows / batchSize;
            IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

            //Input projections
            var weightedQKV = g.View(g.Affine(inputQNorm, QKV, QKVb), dims: new long[] { batchSize, seqLenQ, 3, m_multiHeadNum, m_d });
            var allQ = g.Select(weightedQKV, 2, 0);
            var allK = g.Select(weightedQKV, 2, 1);
            var allV = g.Select(weightedQKV, 2, 2);

            //Multi-head attentions
            IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
            if (m_PEType == PositionEmbeddingEnums.RoPE)
            {
                Qs = g.RoPE(Qs, seqLenQ);
            }

            int q_start_offset = 0;
            int newCacheRowSize = seqLenQ;
            IWeightTensor m_cacheQs = null; //[seqLen, batchSize, m_multiHeadNum * m_d]
            string QKeyName = keyName + "_" + nameof(inputQ);
            if (cachedTensors != null)
            {
                if (cachedTensors.ContainsKey(QKeyName) == true)
                {
                    m_cacheQs = cachedTensors[QKeyName];
                    q_start_offset = (int)m_cacheQs.Sizes[0];
                    newCacheRowSize = seqLenQ - (int)m_cacheQs.Sizes[0];
                }
                else
                {
                    cachedTensors.Add(QKeyName, null);
                }
            }

            Qs = g.View(Qs, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });

            IWeightTensor Ks = null;
            if (m_PEType == PositionEmbeddingEnums.RoPE)
            {
                Ks = g.View(g.AsContiguous(g.Transpose(allK, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                Ks = g.RoPE(Ks, seqLenQ);
                Ks = g.View(Ks, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
            }
            else
            {
                Ks = g.View(g.AsContiguous(g.Transpose(allK, 1, 2)), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
            }

            IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
            IWeightTensor o = g.FlashAttention(Qs, Ks, Vs, q_start_offset); // shape: [batchSize, m_multiHeadNum, seqLen, m_d]

            if (q_start_offset > 0)
            {
                o = g.Peek(o, 2, q_start_offset, newCacheRowSize);
                o = g.AsContiguous(o);
            }

            IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2), false), dims: new long[] { batchSize * newCacheRowSize, m_multiHeadNum * m_d });

            // Output projection
            IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), m_dropoutRatio, inPlace: false); // Shape: [batchSize * relPosSize, m_multiHeadNum * m_d]


            if (cachedTensors != null)
            {
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize, newCacheRowSize, m_multiHeadNum * m_d });
                finalAttResults = g.Transpose(finalAttResults, 0, 1); // Shape: [newCacheRowSize, batchSize, m_multiHeadNum * m_d]

                if (m_cacheQs == null)
                {
                    m_cacheQs = finalAttResults;
                }
                else
                {
                    m_cacheQs = g.Concate(0, m_cacheQs, finalAttResults);
                }
                m_cacheQs.UnbindFromComputeGraph();

                cachedTensors[QKeyName] = m_cacheQs;

                finalAttResults = g.AsContiguous(g.Transpose(m_cacheQs, 0, 1)); // Shape: [batchSize, seqLenQ, m_multiHeadNum * m_d]
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });
            }


            // For runtime, we don't call it by inplace, since it will change the content of cached weights
            IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: false);

            return result;
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor Shape: [batchSize * seqLen, input_dim] </param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public (IWeightTensor, IWeightTensor) Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph, bool outputAttenWeights = false, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            string keyName = $"{m_name}_MultiHeadAttention_3";
            using IComputeGraph g = graph.CreateSubGraph(keyName);
            int seqLenQ = inputQ.Rows / batchSize;

            int newTokensIdx = seqLenQ;
            IWeightTensor m_cacheQs = null;
            string QKeyName = keyName + "_" + nameof(inputQ);
            if (cachedTensors != null)
            {
                if (cachedTensors.ContainsKey(QKeyName) == true)
                {
                    m_cacheQs = cachedTensors[QKeyName];
                    newTokensIdx = seqLenQ - (int)m_cacheQs.Sizes[0];
                }
                else
                {
                    cachedTensors.Add(QKeyName, null);
                }

                // Optimize runtime for test that only processing new tokens
                inputQ = g.View(inputQ, dims: new long[] { batchSize, seqLenQ, -1 });
                inputQ = g.AsContiguous(g.Peek(inputQ, 1, seqLenQ - newTokensIdx, newTokensIdx)); // Shape: [batchSize, newTokensSize, input_dim]
                inputQ = g.View(inputQ, dims: new long[] { batchSize * newTokensIdx, -1 }); // Shape: [batchSize * newTokensSize, input_dim]
            }

            // SeqLenK must be euqal to SeqLenV
            int seqLenK = inputK.Rows / batchSize;
            int seqLenV = inputV.Rows / batchSize;

            IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

            //Input projections
            IWeightTensor allQ = g.View(g.Affine(inputQNorm, Q, Qb), dims: new long[] { batchSize, newTokensIdx, m_multiHeadNum, m_d });

            //Multi-head attentions
            IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, newTokensIdx, m_d });
            IWeightTensor Ks = null;
            IWeightTensor Vs = null;

            if (cachedTensors == null) // We don't use any cached tensors
            {
                IWeightTensor allK = g.View(g.Affine(inputK, K, Kb), dims: new long[] { batchSize, seqLenK, m_multiHeadNum, m_d });
                IWeightTensor allV = g.View(g.Affine(inputV, V, Vb), dims: new long[] { batchSize, seqLenV, m_multiHeadNum, m_d });
                Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenK });
                Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenV, m_d });
            }
            else
            {
                string KsCacheName = keyName + "_" + nameof(Ks);
                string VsCacheName = keyName + "_" + nameof(Vs);

                if (cachedTensors.ContainsKey(KsCacheName) == false)
                {
                    IWeightTensor allK = g.View(g.Affine(inputK, K, Kb), dims: new long[] { batchSize, seqLenK, m_multiHeadNum, m_d });
                    Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenK });

                    cachedTensors.Add(KsCacheName, Ks.CopyWeightsRef(KsCacheName, Ks.NeedGradient, graphToBind: null));
                }
                else
                {
                    Ks = cachedTensors[KsCacheName];
                }

                if (cachedTensors.ContainsKey(VsCacheName) == false)
                {
                    IWeightTensor allV = g.View(g.Affine(inputV, V, Vb), dims: new long[] { batchSize, seqLenV, m_multiHeadNum, m_d });
                    Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenV, m_d });
                    cachedTensors.Add(VsCacheName, Vs.CopyWeightsRef(VsCacheName, Vs.NeedGradient, graphToBind: null));
                }
                else
                {
                    Vs = cachedTensors[VsCacheName];
                }
            }


            // Scaled softmax
            float scale = 1.0f / (float)(Math.Sqrt(m_d));
            var attn = g.MulBatch(Qs, Ks, scale); // Shape: [batchSize * m_multiHeadNum, newTokensIdx, seqLenK]
            attn = g.View(attn, dims: new long[] { batchSize, m_multiHeadNum, newTokensIdx, seqLenK });

            if (keyMask != null)
            {
                if (cachedTensors != null)
                {
                    keyMask = g.Peek(keyMask, 2, seqLenQ - newTokensIdx, newTokensIdx); // Shape: [batchSize, m_multiHeadNum, newTokensIdx, seqLenK]
                }

                attn = g.Add(attn, keyMask, inPlace: true);  // Shape: [batchSize, m_multiHeadNum, newTokensIdx, seqLenK]
            }

            var attnProbs = g.Softmax(attn, inPlace: true);  // Shape: [batchSize, m_multiHeadNum, newTokensIdx, seqLenK]

            IWeightTensor sumAttnWeights = null;
            if (outputAttenWeights)
            {
                sumAttnWeights = g.Select(attnProbs, 1, 0);
                for (int i = 1; i < m_multiHeadNum; i++)
                {
                    var tmp = g.Select(attnProbs, 1, i);
                    sumAttnWeights = g.Add(sumAttnWeights, tmp);
                }

                sumAttnWeights = graph.Div(sumAttnWeights, (float)m_multiHeadNum, inPlace: true);
                sumAttnWeights = graph.View(sumAttnWeights, new long[] { batchSize * newTokensIdx, seqLenK });
            }

            attnProbs = g.View(attnProbs, dims: new long[] { batchSize * m_multiHeadNum, newTokensIdx, seqLenK });

            IWeightTensor o = g.View(g.MulBatch(attnProbs, Vs), dims: new long[] { batchSize, m_multiHeadNum, newTokensIdx, m_d });
            IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * newTokensIdx, m_multiHeadNum * m_d });

            // Output projection
            IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), m_dropoutRatio, inPlace: true);
            IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: true); // Shape: [batchSize * newTokensSize, input_dim]


            if (cachedTensors != null)
            {
                result = g.View(result, dims: new long[] { batchSize, newTokensIdx, m_multiHeadNum * m_d });
                result = g.AsContiguous(g.Transpose(result, 0, 1)); // Shape: [newTokensIdx, batchSize, m_multiHeadNum * m_d]

                if (m_cacheQs == null)
                {
                    m_cacheQs = result;// Shape: [newTokensIdx, batchSize, m_multiHeadNum * m_d]
                }
                else
                {
                    m_cacheQs = g.Concate(0, m_cacheQs, result);
                }
                m_cacheQs.UnbindFromComputeGraph();

                cachedTensors[QKeyName] = m_cacheQs;

                result = g.AsContiguous(g.Transpose(m_cacheQs, 0, 1)); // Shape: [batchSize, seqLenQ, m_multiHeadNum * m_d]
                result = graph.View(result, dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });
            }

            return (result, sumAttnWeights);
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                W0,
                b0
            };

            if (m_sharedQKV == false)
            {
                response.Add(Q);
                response.Add(Qb);

                response.Add(K);
                response.Add(Kb);

                response.Add(V);
                response.Add(Vb);
            }
            else
            {
                response.Add(QKV);
                response.Add(QKVb);
            }

            response.AddRange(layerNormQ.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            if (m_sharedQKV == false)
            {
                Q.Save(stream);
                Qb.Save(stream);

                K.Save(stream);
                Kb.Save(stream);

                V.Save(stream);
                Vb.Save(stream);
            }
            else
            {
                QKV.Save(stream);
                QKVb.Save(stream);
            }

            W0.Save(stream);
            b0.Save(stream);

            layerNormQ.Save(stream);


        }


        public void Load(IModel stream)
        {
            if (m_sharedQKV == false)
            {
                Q.Load(stream);
                Qb.Load(stream);

                K.Load(stream);
                Kb.Load(stream);

                V.Load(stream);
                Vb.Load(stream);
            }
            else
            {
                QKV.Load(stream);
                QKVb.Load(stream);
            }

            W0.Load(stream);
            b0.Load(stream);

            layerNormQ.Load(stream);
        }
    }
}
