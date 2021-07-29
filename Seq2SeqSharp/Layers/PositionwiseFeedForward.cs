using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class PositionwiseFeedForward
    {
        private readonly LayerNormalization layerNorm2;
        private readonly FeedForwardLayer feedForwardLayer1;
        private readonly FeedForwardLayer feedForwardLayer2;

        private readonly string m_name;
        private readonly float m_dropoutRatio;

        public PositionwiseFeedForward(string name, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f)
        {
            m_name = name;
            m_dropoutRatio = dropoutRatio;

            layerNorm2 = new LayerNormalization($"{name}.{nameof(layerNorm2)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor);
            feedForwardLayer1 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer1)}", hiddenDim, hiddenDim * 4, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor);
            feedForwardLayer2 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer2)}", hiddenDim * 4, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor);
        }
      
        public IWeightTensor Perform(IWeightTensor input, int batchSize, IComputeGraph graph)
        {
            using IComputeGraph g = graph.CreateSubGraph($"{m_name}_PositionwiseFeedForward");
            var inputNorm = layerNorm2.Norm(input, g);

            //Feed forward
            IWeightTensor ffnResult = feedForwardLayer1.Process(inputNorm, batchSize, g);
            IWeightTensor reluFFNResult = g.Relu(ffnResult, inPlace: true);
            IWeightTensor ffn2Result = feedForwardLayer2.Process(reluFFNResult, batchSize, g);

            //Skip connection and layer normaliztion
            IWeightTensor addFFNResult = graph.Add(ffn2Result, input, inPlace: true);

            return addFFNResult;

        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(layerNorm2.GetParams());
            response.AddRange(feedForwardLayer1.GetParams());
            response.AddRange(feedForwardLayer2.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            layerNorm2.Save(stream);
            feedForwardLayer1.Save(stream);
            feedForwardLayer2.Save(stream);
        }


        public void Load(IModel stream)
        {
            layerNorm2.Load(stream);
            feedForwardLayer1.Load(stream);
            feedForwardLayer2.Load(stream);
        }
    }
}
