using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class MultiHeadAttention
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

        private readonly LayerNormalization layerNormQ;

        private readonly int m_hiddenDim;
        private readonly int m_d;
        private readonly int m_multiHeadNum;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        private readonly bool m_sharedQKV;

        public MultiHeadAttention(string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId, bool isTrainable, bool sharedQKV = false)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_multiHeadNum = multiHeadNum;
            m_d = m_hiddenDim / m_multiHeadNum;
            m_dropoutRatio = dropoutRatio;
            m_sharedQKV = sharedQKV;

            W0 = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.{nameof(W0)}", isTrainable: isTrainable, normType: NormType.Uniform);
            b0 = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(b0)}", isTrainable: isTrainable);

            if (m_sharedQKV == false)
            {
                Q = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: isTrainable, normType: NormType.Uniform);
                Qb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: isTrainable);

                K = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(K)}", isTrainable: isTrainable, normType: NormType.Uniform);
                Kb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Kb)}", isTrainable: isTrainable);

                V = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(V)}", isTrainable: isTrainable, normType: NormType.Uniform);
                Vb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Vb)}", isTrainable: isTrainable);
            }
            else
            {
                QKV = new WeightTensor(new long[2] { inputDim, hiddenDim * 3 }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: isTrainable, normType: NormType.Uniform);
                QKVb = new WeightTensor(new long[2] { 1, hiddenDim * 3 }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: isTrainable);
            }

            layerNormQ = new LayerNormalization($"{name}.{nameof(layerNormQ)}", m_hiddenDim, deviceId, isTrainable);
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public (IWeightTensor, IWeightTensor) Perform(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph, bool outputAttenWeights = false)
        {
            using (IComputeGraph g = graph.CreateSubGraph($"{m_name}_MultiHeadAttention"))
            {
                int seqLenQ = inputQ.Rows / batchSize;

                IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

                //Input projections
                var weightedQKV = g.View(g.Affine(inputQNorm, QKV, QKVb), dims: new long[] {batchSize, seqLenQ, 3, m_multiHeadNum, m_d });
                var allQ = g.Select(weightedQKV, 2, 0);
                var allK = g.Select(weightedQKV, 2, 1);
                var allV = g.Select(weightedQKV, 2, 2);


                //Multi-head attentions
                IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenQ });
                IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });

                // Scaled softmax
                float scale = 1.0f / (float)(Math.Sqrt(m_d));
                var attn = g.MulBatch(Qs, Ks, scale);
                attn = g.View(attn, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenQ });

                if (keyMask != null)
                {
                    attn = g.Add(attn, keyMask, runGradient1: true, runGradient2: false, inPlace: true);
                }

                var attnProbs = g.Softmax(attn, inPlace: true);

                IWeightTensor sumAttnWeights = null;
                if (outputAttenWeights)
                {
                    //Merge all attention probs over multi-heads
                    sumAttnWeights = graph.Sum(attnProbs, 1, runGradient: false);
                    sumAttnWeights = graph.View(sumAttnWeights, false, new long[] { batchSize, seqLenQ, seqLenQ });
                }

                attnProbs = g.View(attnProbs, dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, seqLenQ });

                IWeightTensor o = g.View(g.MulBatch(attnProbs, Vs), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });

                // Output projection
                IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), batchSize, m_dropoutRatio, inPlace: true);
                IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: true);


                return (result, sumAttnWeights);
            }
        }


        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public (IWeightTensor, IWeightTensor) Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph, bool outputAttenWeights = false)
        {
            using (IComputeGraph g = graph.CreateSubGraph($"{m_name}_MultiHeadAttention"))
            {
                int seqLenQ = inputQ.Rows / batchSize;

                // SeqLenK must be euqal to SeqLenV
                int seqLenK = inputK.Rows / batchSize;
                int seqLenV = inputV.Rows / batchSize;

                IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);

                //Input projections
                IWeightTensor allQ = g.View(g.Affine(inputQNorm, Q, Qb), dims: new long[] { batchSize, seqLenQ, m_multiHeadNum, m_d });
                IWeightTensor allK = g.View(g.Affine(inputK, K, Kb), dims: new long[] { batchSize, seqLenK, m_multiHeadNum, m_d });
                IWeightTensor allV = g.View(g.Affine(inputV, V, Vb), dims: new long[] { batchSize, seqLenV, m_multiHeadNum, m_d });

                //Multi-head attentions
                IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenK });
                IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenV, m_d });

                // Scaled softmax
                float scale = 1.0f / (float)(Math.Sqrt(m_d));
                var attn = g.MulBatch(Qs, Ks, scale);
                attn = g.View(attn, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenK });

                if (keyMask != null)
                {
                    attn = g.Add(attn, keyMask, runGradient1: true, runGradient2: false, inPlace: true);
                }

                var attnProbs = g.Softmax(attn, inPlace: true);

                IWeightTensor sumAttnWeights = null;
                if (outputAttenWeights)
                {
                    //Merge all attention probs over multi-heads
                    sumAttnWeights = g.Sum(attnProbs, 1, runGradient: false);
                    sumAttnWeights = graph.Mul(sumAttnWeights, 1.0f / (float)m_multiHeadNum, runGradient: false);
                    sumAttnWeights = graph.View(sumAttnWeights, false, new long[] { batchSize * seqLenQ, seqLenK });
                }

                attnProbs = g.View(attnProbs, dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, seqLenK });

                IWeightTensor o = g.View(g.MulBatch(attnProbs, Vs), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });

                // Output projection
                IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), batchSize, m_dropoutRatio, inPlace: true);
                IWeightTensor result = graph.Add(finalAttResults, inputQ, inPlace: true);

                
                return (result, sumAttnWeights);
            }
        }

        public virtual List<IWeightTensor> getParams()
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

            response.AddRange(layerNormQ.getParams());

            return response;
        }


        public void Save(Stream stream)
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


        public void Load(Stream stream)
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
