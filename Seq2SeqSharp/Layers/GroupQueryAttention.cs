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
    internal class GroupQueryAttention : IAttentionLayer
    {
        private readonly IWeightTensor W0;
        private readonly IWeightTensor b0;

        private readonly IWeightTensor Q;
        private readonly IWeightTensor K;
        private readonly IWeightTensor V;

        private readonly IWeightTensor Qb;
        private readonly IWeightTensor Kb;
        private readonly IWeightTensor Vb;

        private readonly INormalization layerNormQ;

        private readonly int m_d_out;
        private readonly int m_head_dim;
        private readonly int m_num_heads;
        private readonly int m_num_kv_groups;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        private readonly PositionEmbeddingEnums m_PEType;

        public GroupQueryAttention(string name, int num_heads, int num_kv_groups, int d_out, int d_in, float dropoutRatio, int deviceId, bool isTrainable,
            float learningRateFactor = 1.0f, DType elementType = DType.Float32, PositionEmbeddingEnums peType = PositionEmbeddingEnums.APE, NormEnums normType = NormEnums.LayerNorm)
        {
            m_name = name;
            m_d_out = d_out;
            m_num_heads = num_heads;
            m_num_kv_groups = num_kv_groups;

            if (num_kv_groups <= 0)
            {
                throw new ArgumentException("The number of KV groups must be greater than 0.");
            }

            if (m_d_out % m_num_heads != 0)
            {
                throw new ArgumentException("The hidden dim must be divisible by multi-head size.");
            }

            m_head_dim = m_d_out / m_num_heads;

            if (m_num_heads % m_num_kv_groups != 0)
            {
                throw new ArgumentException("The number of heads must be divisible by the number of KV groups");
            }

            m_dropoutRatio = dropoutRatio;
            m_PEType = peType;

            Logger.WriteLine(Logger.Level.debug, $"Creating multi-head attention layer. Name = '{name}', HiddenDim = '{d_out}', multi-head dim = '{num_heads}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', PE = '{peType}', Norm = '{normType}'");

            W0 = new WeightTensor(new long[2] { d_out, d_out }, deviceId, name: $"{name}.{nameof(W0)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
            b0 = new WeightTensor(new long[2] { 1, d_out }, 0, deviceId, name: $"{name}.{nameof(b0)}", isTrainable: isTrainable, dtype: elementType);

            Q = new WeightTensor(new long[2] { d_in, d_out }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
            Qb = new WeightTensor(new long[2] { 1, d_out }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

            K = new WeightTensor(new long[2] { d_in, m_num_kv_groups * m_head_dim }, deviceId, name: $"{name}.{nameof(K)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
            Kb = new WeightTensor(new long[2] { 1, m_num_kv_groups * m_head_dim }, 0, deviceId, name: $"{name}.{nameof(Kb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

            V = new WeightTensor(new long[2] { d_in, m_num_kv_groups * m_head_dim }, deviceId, name: $"{name}.{nameof(V)}", isTrainable: isTrainable, initType: RandomInitType.Uniform, learningRateFactor: learningRateFactor, dtype: elementType);
            Vb = new WeightTensor(new long[2] { 1, m_num_kv_groups * m_head_dim }, 0, deviceId, name: $"{name}.{nameof(Vb)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);


            if (normType == NormEnums.LayerNorm)
            {
                layerNormQ = new LayerNormalization($"{name}.{nameof(layerNormQ)}", m_d_out, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
            else
            {
                layerNormQ = new RMSNormalization($"{name}.{nameof(layerNormQ)}", m_d_out, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
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
            throw new NotImplementedException();
        }
       
        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            string keyName = $"{m_name}_GroupQueryAttention_1";
            using IComputeGraph g = graph.CreateSubGraph(keyName);
            int seqLenQ = inputQ.Rows / batchSize;
            IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

            //Input projections
            var allQ = g.View(g.Affine(inputQNorm, Q, Qb), dims: new long[] { batchSize, seqLenQ, m_num_heads, m_head_dim });
            var allK = g.View(g.Affine(inputQNorm, K, Kb), dims: new long[] { batchSize, seqLenQ, m_num_kv_groups, m_head_dim });
            var allV = g.View(g.Affine(inputQNorm, V, Vb), dims: new long[] { batchSize, seqLenQ, m_num_kv_groups, m_head_dim });

            //Multi-head attentions
            IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_num_heads, seqLenQ, m_head_dim });
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

            int group_size = m_num_heads / m_num_kv_groups;
            IWeightTensor Ks = null;
            if (m_PEType == PositionEmbeddingEnums.RoPE)
            {
                Ks = g.View(g.AsContiguous(g.Transpose(allK, 1, 2)), dims: new long[] { batchSize * m_num_kv_groups, seqLenQ, m_head_dim });
                Ks = g.RoPE(Ks, seqLenQ);
            }
            else
            {
                Ks = g.AsContiguous(g.Transpose(allK, 1, 2)); //Shape: [batchSize, m_num_kv_groups, seqLenQ, m_head_dim]
            }
            Ks = g.View(Ks, dims: new long[] { batchSize, m_num_kv_groups, 1, seqLenQ, m_head_dim });
            Ks = g.Expand(Ks, dims: new long[] { batchSize, m_num_kv_groups, group_size, seqLenQ, m_head_dim });
            Ks = g.Transpose(Ks, 3, 4); //Shape: [batchSize, m_num_kv_groups, group_size, m_head_dim, seqLenQ]
            Ks = g.View(g.AsContiguous(Ks), dims: new long[] { batchSize * m_num_heads, m_head_dim, seqLenQ });


            IWeightTensor Vs = g.AsContiguous(g.Transpose(allV, 1, 2)); //Shape: [batchSize, m_num_kv_groups, seqLenQ, m_head_dim]
            Vs = g.View(Vs, dims: new long[] { batchSize, m_num_kv_groups, 1, seqLenQ, m_head_dim });
            Vs = g.Expand(Vs, dims: new long[] { batchSize, m_num_kv_groups, group_size, seqLenQ, m_head_dim });
            Vs = g.View(g.AsContiguous(Vs), dims: new long[] { batchSize * m_num_heads, seqLenQ, m_head_dim });


            // Scaled masked softmax
            float scale = 1.0f / (float)(Math.Sqrt(m_head_dim));

            // Convert tensors to Float32 type if they are not that type.
            bool useF16 = (Qs.ElementType == DType.Float16);
            if (useF16)
            {
                Qs = g.Half2Float(Qs);
                Ks = g.Half2Float(Ks);
            }
            var attn = g.MulBatch(Qs, Ks, scale); // Shape: [batchSize * m_multiHeadNum, relPosSize, seqLenQ]

            // Add mask
            attn = g.View(attn, dims: new long[] { batchSize, m_num_heads, newTokensIdx, seqLenQ });
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

            attn = g.View(attn, dims: new long[] { batchSize * m_num_heads, newTokensIdx, seqLenQ });
            IWeightTensor o = g.View(g.MulBatch(attn, Vs), dims: new long[] { batchSize, m_num_heads, newTokensIdx, m_head_dim });
            IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * newTokensIdx, m_num_heads * m_head_dim });

            // Output projection
            IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), m_dropoutRatio, inPlace: true); // Shape: [batchSize * relPosSize, m_multiHeadNum * m_d]

            if (cachedTensors != null)
            {
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize, newTokensIdx, m_num_heads * m_head_dim });
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
                finalAttResults = g.View(finalAttResults, dims: new long[] { batchSize * seqLenQ, m_num_heads * m_head_dim });
            }

            // For runtime, we don't call it by inplace, since it will change the content of cached weights
            IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: true);

            return result;
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                W0,
                b0
            };

            response.Add(Q);
            response.Add(Qb);

            response.Add(K);
            response.Add(Kb);

            response.Add(V);
            response.Add(Vb);

            response.AddRange(layerNormQ.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            Q.Save(stream);
            Qb.Save(stream);

            K.Save(stream);
            Kb.Save(stream);

            V.Save(stream);
            Vb.Save(stream);

            W0.Save(stream);
            b0.Save(stream);

            layerNormQ.Save(stream);

        }

        public void Load(IModel stream)
        {
            Q.Load(stream);
            Qb.Load(stream);

            K.Load(stream);
            Kb.Load(stream);

            V.Load(stream);
            Vb.Load(stream);

            W0.Load(stream);
            b0.Load(stream);

            layerNormQ.Load(stream);
        }
    }
}
