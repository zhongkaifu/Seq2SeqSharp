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

        private readonly LayerNormalization layerNormQ;

        private readonly int m_hiddenDim;
        private readonly int m_d;
        private readonly int m_multiHeadNum;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        public MultiHeadAttention(string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId, bool isTrainable)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_multiHeadNum = multiHeadNum;
            m_d = m_hiddenDim / m_multiHeadNum;
            m_dropoutRatio = dropoutRatio;

            W0 = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.{nameof(W0)}", isTrainable: isTrainable, normType: NormType.Uniform);
            b0 = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(b0)}", isTrainable: isTrainable);

            Q = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: isTrainable, normType: NormType.Uniform);
            Qb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: isTrainable);

            K = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(K)}", isTrainable: isTrainable, normType: NormType.Uniform);
            Kb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Kb)}", isTrainable: isTrainable);

            V = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(V)}", isTrainable: isTrainable, normType: NormType.Uniform);
            Vb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Vb)}", isTrainable: isTrainable);

            layerNormQ = new LayerNormalization($"{name}.{nameof(layerNormQ)}", m_hiddenDim, deviceId, isTrainable);
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph)
        {
            using (IComputeGraph g = graph.CreateSubGraph($"{m_name}_MultiHeadAttention"))
            {
                int seqLenQ = inputQ.Rows / batchSize;

                // SeqLenK must be euqal to SeqLenV
                int seqLenK = inputK.Rows / batchSize;
                int seqLenV = inputV.Rows / batchSize;

                IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);
                if (inputK == inputQ)
                {
                    inputK = inputQNorm;
                }
                if (inputV == inputQ)
                {
                    inputV = inputQNorm;
                }


                //Input projections
                float scale = 1.0f;
                IWeightTensor allQ = g.View(g.Affine(inputQNorm, Q, Qb, scale), dims: new long[] { batchSize, seqLenQ, m_multiHeadNum, m_d });
                IWeightTensor allK = g.View(g.Affine(inputK, K, Kb, scale), dims: new long[] { batchSize, seqLenK, m_multiHeadNum, m_d });
                IWeightTensor allV = g.View(g.Affine(inputV, V, Vb, scale), dims: new long[] { batchSize, seqLenV, m_multiHeadNum, m_d });

                //Multi-head attentions
                IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenK });
                IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenV, m_d });

                // Scaled softmax
                scale = 1.0f / (float)(Math.Sqrt(m_d));
                IWeightTensor attn = g.MulBatch(Qs, Ks, batchSize * m_multiHeadNum, scale);


                if (keyMask != null)
                {
                    using (var keyMaskView = g.View(keyMask, runGradient: false, dims: new long[] { batchSize, 1, seqLenQ, seqLenK }))
                    {
                        using (var keyMaskViewExp = g.Expand(keyMaskView, runGradient: false, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenK }))
                        {
                            using (var keyMaskViewExpConti = g.AsContiguous(keyMaskViewExp, runGradient: false))
                            {
                                using (var keyMaskViewExpContiView = g.View(keyMaskViewExpConti, runGradient: false, dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, seqLenK }))
                                {
                                    attn = g.Add(attn, keyMaskViewExpContiView, runGradient1: true, runGradient2: false);
                                }
                            }
                        }
                    }
                }

                IWeightTensor softmax = g.Softmax(attn, inPlace: true);

                IWeightTensor o = g.View(g.MulBatch(softmax, Vs, batchSize * m_multiHeadNum), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });

                IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });

                // Output projection
                IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), batchSize, m_dropoutRatio, inPlace: true);

                return graph.Add(finalAttResults, inputQ);
            }
        }

        public virtual List<IWeightTensor> getParams()
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

            response.AddRange(layerNormQ.getParams());

            return response;
        }


        public void Save(Stream stream)
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


        public void Load(Stream stream)
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
