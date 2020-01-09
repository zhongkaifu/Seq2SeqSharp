using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class SelfAttention
    {
        private readonly IWeightTensor W0;
        private readonly IWeightTensor b0;

        private readonly IWeightTensor Q;
        private readonly IWeightTensor K;
        private readonly IWeightTensor V;

        private readonly IWeightTensor Qb;
        private readonly IWeightTensor Kb;
        private readonly IWeightTensor Vb;

        private readonly LayerNormalization layerNorm1;
        private readonly LayerNormalization layerNorm2;
        private readonly FeedForwardLayer feedForwardLayer1;
        private readonly FeedForwardLayer feedForwardLayer2;

        private readonly int m_hiddenDim;
        private readonly int m_d;
        private readonly int m_multiHeadNum;
        private readonly string m_name;
        private readonly float m_dropoutRatio;

        public SelfAttention(string name, int multiHeadNum, int hiddenDim, int inputDim, float dropoutRatio, int deviceId)
        {
            m_name = name;
            m_hiddenDim = hiddenDim;
            m_multiHeadNum = multiHeadNum;
            m_d = m_hiddenDim / m_multiHeadNum;
            m_dropoutRatio = dropoutRatio;

            W0 = new WeightTensor(new long[2] { hiddenDim, hiddenDim }, deviceId, name: $"{name}.{nameof(W0)}", isTrainable: true);
            b0 = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(b0)}", isTrainable: true);

            Q = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(Q)}", isTrainable: true);
            Qb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Qb)}", isTrainable: true);

            K = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(K)}", isTrainable: true);
            Kb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Kb)}", isTrainable: true);

            V = new WeightTensor(new long[2] { inputDim, hiddenDim }, deviceId, name: $"{name}.{nameof(V)}", isTrainable: true);
            Vb = new WeightTensor(new long[2] { 1, hiddenDim }, 0, deviceId, name: $"{name}.{nameof(Vb)}", isTrainable: true);


            layerNorm1 = new LayerNormalization($"{name}.{nameof(layerNorm1)}", hiddenDim, deviceId);
            layerNorm2 = new LayerNormalization($"{name}.{nameof(layerNorm2)}", hiddenDim, deviceId);
            feedForwardLayer1 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer1)}", hiddenDim, hiddenDim * 4, m_dropoutRatio, deviceId);
            feedForwardLayer2 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer2)}", hiddenDim * 4, hiddenDim, m_dropoutRatio, deviceId);
        }

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="g">The instance of computing graph</param>
        /// <returns></returns>
        public IWeightTensor Perform(IWeightTensor input, int batchSize, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name);
            int seqLen = input.Rows / batchSize;
            IWeightTensor nInput = layerNorm1.Norm(input, g);

            //Input projections
            IWeightTensor allQ = g.View(g.Affine(nInput, Q, Qb), batchSize, seqLen, m_multiHeadNum, m_d);
            IWeightTensor allK = g.View(g.Affine(nInput, K, Kb), batchSize, seqLen, m_multiHeadNum, m_d);
            IWeightTensor allV = g.View(g.Affine(nInput, V, Vb), batchSize, seqLen, m_multiHeadNum, m_d);

            //Multi-head attentions
            IWeightTensor Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), m_multiHeadNum * batchSize, seqLen, m_d);
            IWeightTensor Ks = g.View(g.Permute(allK, 2, 0, 3, 1), m_multiHeadNum * batchSize, m_d, seqLen);
            IWeightTensor Vs = g.View(g.Permute(allV, 2, 0, 1, 3), m_multiHeadNum * batchSize, seqLen, m_d);

            // Scaled softmax
            float scale = 1.0f / (float)Math.Sqrt(m_d);
            IWeightTensor attn = g.MulBatch(Qs, Ks, m_multiHeadNum * batchSize, scale);
            IWeightTensor attn2 = g.View(attn, m_multiHeadNum * batchSize * seqLen, seqLen);

            IWeightTensor softmax = g.Softmax(attn2, inPlace: true);
            IWeightTensor softmax2 = g.View(softmax, m_multiHeadNum * batchSize, seqLen, seqLen);
            IWeightTensor o = g.View(g.MulBatch(softmax2, Vs, m_multiHeadNum * batchSize), m_multiHeadNum, batchSize, seqLen, m_d);
            IWeightTensor W = g.View(g.Permute(o, 1, 2, 0, 3), batchSize * seqLen, m_multiHeadNum * m_d);

            // Output projection
            IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), batchSize, m_dropoutRatio, inPlace: true);

            //Skip connection and layer normaliztion
            IWeightTensor normAddedAttResult = layerNorm2.AddNorm(finalAttResults, input, g);

            //Feed forward
            IWeightTensor ffnResult = feedForwardLayer1.Process(normAddedAttResult, batchSize, g);
            IWeightTensor reluFFNResult = g.Relu(ffnResult);
            IWeightTensor ffn2Result = feedForwardLayer2.Process(reluFFNResult, batchSize, g);

            //Skip connection and layer normaliztion
            IWeightTensor addFFNResult = g.Add(ffn2Result, normAddedAttResult);

            return addFFNResult;
        }


        public virtual List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                Q,
                Qb,

                K,
                Kb,

                V,
                Vb,

                W0,
                b0
            };

            response.AddRange(layerNorm1.getParams());
            response.AddRange(layerNorm2.getParams());
            response.AddRange(feedForwardLayer1.GetParams());
            response.AddRange(feedForwardLayer2.GetParams());

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

            layerNorm1.Save(stream);
            layerNorm2.Save(stream);
            feedForwardLayer1.Save(stream);
            feedForwardLayer2.Save(stream);
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

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
            feedForwardLayer1.Load(stream);
            feedForwardLayer2.Load(stream);
        }
    }
}
