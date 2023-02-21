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
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    internal class PositionwiseFeedForward : IFeedForwardLayer
    {
        private readonly LayerNormalization layerNorm2;
        private readonly FeedForwardLayer feedForwardLayer1;
        private readonly FeedForwardLayer feedForwardLayer2;

        private readonly string m_name;
        private readonly float m_dropoutRatio;

        private ActivateFuncEnums m_activateFunc;

        public PositionwiseFeedForward(string name, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.Relu)
        {
            m_name = name;
            m_dropoutRatio = dropoutRatio;
            m_activateFunc = activateFunc;

            Logger.WriteLine($"Creating positionwise feed forward layer. Name = '{name}', HiddenDim = '{hiddenDim}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', Activate Function = '{activateFunc}'");

            layerNorm2 = new LayerNormalization($"{name}.{nameof(layerNorm2)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor);
            feedForwardLayer1 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer1)}", hiddenDim, hiddenDim * 4, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor);
            feedForwardLayer2 = new FeedForwardLayer($"{name}.{nameof(feedForwardLayer2)}", hiddenDim * 4, hiddenDim, m_dropoutRatio, deviceId, isTrainable, learningRateFactor: learningRateFactor);
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

            int seqLen = (int)(input.Sizes[0] / batchSize);


            IWeightTensor m_cacheT = null;
            string cacheKeyName = keyName + "_" + nameof(input);
            int newTokensIdx = seqLen;
            if (cachedTensors != null)
            {
                if (cachedTensors.ContainsKey(cacheKeyName) == true)
                {
                    m_cacheT = cachedTensors[cacheKeyName];
                    newTokensIdx = seqLen - (int)m_cacheT.Sizes[0];
                }
                else
                {
                    cachedTensors.Add(cacheKeyName, null);
                }


                // Optimize runtime for test that only processing new tokens
                input = g.View(input, dims: new long[] { batchSize, seqLen, -1 });

                input = g.AsContiguous(g.Peek(input, 1, seqLen - newTokensIdx, newTokensIdx)); // Shape: [batchSize, newTokenIdx, input_dim]
                input = g.View(input, dims: new long[] { batchSize * newTokensIdx, -1 }); // Shape: [batchSize * newTokenIdx, input_dim]
                                                                                        
            }

            var inputNorm = layerNorm2.Norm(input, g);
            //Feed forward
            var ffnResult = feedForwardLayer1.Process(inputNorm, batchSize, g);
            // Activate function
            var reluFFNResult = ((m_activateFunc == ActivateFuncEnums.Swish) ? g.Swish(ffnResult, inPlace: true) : g.Relu(ffnResult, inPlace: true));
            var ffn2Result = feedForwardLayer2.Process(reluFFNResult, batchSize, g); // Shape: [batchSize * newTokenIdx, input_dim]
            //Skip connection and layer normaliztion
            var addFFNResult = graph.Add(ffn2Result, input, inPlace: true); // Shape: [batchSize * newTokenIdx, input_dim]

            if (cachedTensors != null)
            {
                addFFNResult = g.View(addFFNResult, dims: new long[] {batchSize, newTokensIdx, -1 }); // Shape: [batchSize, newTokenIdx, input_dim]
                addFFNResult = g.Transpose(addFFNResult, 0, 1); // Shape: [newTokenIdx, batchSize, input_dim]

                if (m_cacheT == null)
                {
                    m_cacheT = addFFNResult;// Shape: [newTokenIdx, batchSize, input_dim]
                }
                else
                {
                    m_cacheT = g.Concate(0, m_cacheT, addFFNResult); // Shape: [seqLen, batchSize, input_dim]
                }
                m_cacheT.UnbindFromComputeGraph();

                cachedTensors[cacheKeyName] = m_cacheT;

                addFFNResult = g.AsContiguous(g.Transpose(m_cacheT, 0, 1)); // Shape: [batchSize, seqLen, input_dim]
                addFFNResult = graph.View(addFFNResult, dims: new long[] { batchSize * seqLen, -1 });
            }
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

            stream.AddWeights($"{m_name}.ActivateFunc", new float[1] { (float)m_activateFunc});
        }


        public void Load(IModel stream)
        {
            layerNorm2.Load(stream);
            feedForwardLayer1.Load(stream);
            feedForwardLayer2.Load(stream);

            m_activateFunc = (ActivateFuncEnums)stream.GetWeights($"{m_name}.ActivateFunc")[0];
            Logger.WriteLine($"Loading '{m_name}' activate function setting '{m_activateFunc}'");

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
