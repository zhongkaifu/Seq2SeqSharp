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
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp
{
    internal class PositionwiseFeedForward : IFeedForwardLayer
    {
        private readonly INormalization layerNorm2;
        private readonly FeedForwardLayer feedForwardLayer1 = null;
        private readonly FeedForwardLayer feedForwardLayer2 = null;

        private readonly FeedForwardLayer feedForwardLayer3 = null;

        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly DType m_elementType;

        private ActivateFuncEnums m_activateFunc;

        public PositionwiseFeedForward(string name, int hiddenDim, int intermediateDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.ReLU, DType elementType = DType.Float32, NormEnums normType = NormEnums.LayerNorm)
        {
            m_name = name;
            m_dropoutRatio = dropoutRatio;
            m_activateFunc = activateFunc;
            m_elementType= elementType;

            Logger.WriteLine(Logger.Level.debug, $"Creating positionwise feed forward layer. Name = '{name}', HiddenDim = '{hiddenDim}', IntermediateDim = '{intermediateDim}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', Activate Function = '{activateFunc}', Norm = '{normType}'");

            if (normType == NormEnums.LayerNorm)
            {
                layerNorm2 = new LayerNormalization($"{name}.{nameof(layerNorm2)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }
            else
            {
                layerNorm2 = new RMSNormalization($"{name}.{nameof(layerNorm2)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }

            feedForwardLayer1 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer1)}", hiddenDim, intermediateDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            feedForwardLayer2 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer2)}", intermediateDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);

            if (m_activateFunc == ActivateFuncEnums.SwiGLU)
            {
                feedForwardLayer3 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer3)}", hiddenDim, intermediateDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);
            }

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">input tensor. Shape: [batchSize * seqLen, input_dim]</param>
        /// <param name="batchSize">The batch size of input tensor</param>
        /// <param name="graph">Computing graph</param>
        /// <param name="optmizedRuntime">Enable or not that using optmized runtime in order to reduce computation cost</param>
        /// <returns></returns>
        public IWeightTensor Process(IWeightTensor input, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            string keyName = $"{m_name}_PositionwiseFeedForward";
            using var g = graph.CreateSubGraph(keyName);

            var inputNorm = layerNorm2.Norm(input, g);

            //Feed forward
            var ffnResult = feedForwardLayer1.Process(inputNorm, batchSize, g);
            // Activate function
            var actFFNResult = RunActivateFunction(g, ffnResult);


            if (m_activateFunc == ActivateFuncEnums.SwiGLU)
            {
                var ffnResult2 = feedForwardLayer3.Process(inputNorm, batchSize, g);
                actFFNResult = g.EltMul(actFFNResult, ffnResult2);
            }

            var ffn2Result = feedForwardLayer2.Process(actFFNResult, batchSize, g); // Shape: [batchSize * newTokenIdx, input_dim]

            //Skip connection and layer normaliztion
            var addFFNResult = graph.Add(ffn2Result, input, inPlace: true); // Shape: [batchSize * newTokenIdx, input_dim]

            return addFFNResult;

        }

        private IWeightTensor RunActivateFunction(IComputeGraph g, IWeightTensor tokenEmbs)
        {
            if (m_activateFunc == ActivateFuncEnums.SiLU)
            {
                tokenEmbs = g.SiLU(tokenEmbs, inPlace: true);
            }
            else if (m_activateFunc == ActivateFuncEnums.ReLU)
            {
                tokenEmbs = g.ReLU(tokenEmbs, inPlace: true);
            }
            else if (m_activateFunc == ActivateFuncEnums.LeakyReLU)
            {
                tokenEmbs = g.LeakyReLU(tokenEmbs, inPlace: true);
            }

            return tokenEmbs;
        }


        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(layerNorm2.GetParams());
            response.AddRange(feedForwardLayer1.GetParams());
            response.AddRange(feedForwardLayer2.GetParams());

            if (feedForwardLayer3 != null)
            {
                response.AddRange(feedForwardLayer3.GetParams());
            }

            return response;
        }


        public void Save(IModel stream)
        {
            layerNorm2.Save(stream);
            feedForwardLayer1.Save(stream);
            feedForwardLayer2.Save(stream);

            if (feedForwardLayer3 != null)
            {
                feedForwardLayer3.Save(stream);
            }
        }


        public void Load(IModel stream)
        {
            layerNorm2.Load(stream);
            feedForwardLayer1.Load(stream);
            feedForwardLayer2.Load(stream);

            if (feedForwardLayer3 != null)
            {
                feedForwardLayer3.Load(stream);
            }
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            throw new System.NotImplementedException();
        }

        public int GetDeviceId()
        {
            throw new System.NotImplementedException();
        }
    }
}
