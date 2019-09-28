using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    class SelfAttention
    {
        private IWeightTensor W0;
        private IWeightTensor b0;

        private FeedForwardLayer Q;
        private FeedForwardLayer K;
        private FeedForwardLayer V;

        private LayerNormalization layerNorm1;
        private LayerNormalization layerNorm2;
        private FeedForwardLayer feedForwardLayer1;
        private FeedForwardLayer feedForwardLayer2;

        private int m_batchSize;
        private int m_hiddenDim;
        private int m_d;
        private int m_multiHeadNum;

        public SelfAttention(int batchSize, int multiHeadNum, int hiddenDim, int inputDim, int deviceId)
        {
            m_batchSize = batchSize;
            m_hiddenDim = hiddenDim;
            m_multiHeadNum = multiHeadNum;
            m_d = m_hiddenDim / m_multiHeadNum;

            W0 = new WeightTensor(hiddenDim, hiddenDim, deviceId);
            b0 = new WeightTensor(1, hiddenDim, 0, deviceId);

            Q = new FeedForwardLayer(inputDim, hiddenDim, deviceId);
            K = new FeedForwardLayer(inputDim, hiddenDim, deviceId);
            V = new FeedForwardLayer(inputDim, hiddenDim, deviceId);

            layerNorm1 = new LayerNormalization(hiddenDim, deviceId);
            layerNorm2 = new LayerNormalization(hiddenDim, deviceId);
            feedForwardLayer1 = new FeedForwardLayer(hiddenDim, hiddenDim * 4, deviceId);
            feedForwardLayer2 = new FeedForwardLayer(hiddenDim * 4, hiddenDim, deviceId);
        }       

        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="g">The instance of computing graph</param>
        /// <returns></returns>
        public IWeightTensor Perform(IWeightTensor input, IComputeGraph g)
        {
            var seqLen = input.Rows / m_batchSize;

            //Input projections
            var allQ = g.View(Q.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);
            var allK = g.View(K.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);
            var allV = g.View(V.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);

            //Multi-head attentions
            var Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);
            var Ks = g.View(g.Permute(allK, 2, 0, 3, 1), m_multiHeadNum * m_batchSize, m_d, seqLen);
            var Vs = g.View(g.Permute(allV, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);

            // Scaled softmax
            float scale = 1.0f / (float)Math.Sqrt(m_d);
            var attn = g.MulBatch(Qs, Ks, m_multiHeadNum * m_batchSize, scale);
            var attn2 = g.View(attn, m_multiHeadNum * m_batchSize * seqLen, seqLen);

            var softmax = g.Softmax(attn2);
            var softmax2 = g.View(softmax, m_multiHeadNum * m_batchSize, seqLen, seqLen);
            var o = g.View(g.MulBatch(softmax2, Vs, m_multiHeadNum * m_batchSize), m_multiHeadNum, m_batchSize, seqLen, m_d);
            var W = g.View(g.Permute(o, 1, 2, 0, 3), m_batchSize * seqLen, m_multiHeadNum * m_d);

            // Output projection
            var b0s = g.RepeatRows(b0, W.Rows);
            var finalAttResults = g.MulAdd(W, W0, b0s);

            //Skip connection and layer normaliztion
            var addedAttResult = g.Add(finalAttResults, input);
            var normAddedAttResult = layerNorm1.Process(addedAttResult, g);

            //Feed forward
            var ffnResult = feedForwardLayer1.Process(normAddedAttResult, g);
            var reluFFNResult = g.Relu(ffnResult);
            var ffn2Result = feedForwardLayer2.Process(reluFFNResult, g);

            //Skip connection and layer normaliztion
            var addFFNResult = g.Add(ffn2Result, normAddedAttResult);
            var normAddFFNResult = layerNorm2.Process(addFFNResult, g);

            return normAddFFNResult;
        }


        public virtual List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(Q.GetParams());
            response.AddRange(K.GetParams());
            response.AddRange(V.GetParams());

            response.Add(W0);
            response.Add(b0);

            response.AddRange(layerNorm1.getParams());
            response.AddRange(layerNorm2.getParams());
            response.AddRange(feedForwardLayer1.GetParams());
            response.AddRange(feedForwardLayer2.GetParams());

            return response;
        }


        public void Save(Stream stream)
        {
            Q.Save(stream);
            K.Save(stream);
            V.Save(stream);


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
            K.Load(stream);
            V.Load(stream);

            W0.Load(stream);
            b0.Load(stream);

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
            feedForwardLayer1.Load(stream);
            feedForwardLayer2.Load(stream);
        }
    }
}
