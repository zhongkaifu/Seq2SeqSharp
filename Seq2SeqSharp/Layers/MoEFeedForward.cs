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
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;

namespace Seq2SeqSharp.Layers
{
    internal class MoEFeedForward : IFeedForwardLayer
    {
        private readonly LayerNormalization layerNorm;

        private readonly IWeightTensor m_Whd1;
        private readonly IWeightTensor m_Whd2;

        private readonly IWeightTensor m_Router;
        private readonly IWeightTensor m_RouterBias;

        private readonly string m_name;
        private readonly int m_expertNum;
        private readonly int m_hiddenDim;
        private readonly int m_expertsPerTokenFactor;

        private ActivateFuncEnums m_activateFunc;

        public MoEFeedForward(string name, int expertNum, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.ReLU, int expertsPerTokenFactor = 1, DType elementType = DType.Float32)
        {
            m_name = name;
            m_activateFunc = activateFunc;
            m_expertNum = expertNum;
            m_hiddenDim = hiddenDim;
            m_expertsPerTokenFactor = expertsPerTokenFactor;

            Logger.WriteLine($"Creating MoE feed forward layer. Name = '{name}', ExpertNum = '{expertNum}', ExpertsPerToken = '{expertsPerTokenFactor}', HiddenDim = '{hiddenDim}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', Activate Function = '{activateFunc}'");

            layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor, elementType: elementType);

            m_Whd1 = new WeightTensor(new long[3] { expertNum, hiddenDim, hiddenDim * 4 }, deviceId, name: $"{name}.{nameof(m_Whd1)}", initType: RandomInitType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_Whd2 = new WeightTensor(new long[3] { expertNum, hiddenDim * 4, hiddenDim }, deviceId, name: $"{name}.{nameof(m_Whd2)}", initType: RandomInitType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

            m_Router = new WeightTensor(new long[2] { hiddenDim, expertNum }, deviceId, name: $"{name}.{nameof(m_Router)}", initType: RandomInitType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_RouterBias = new WeightTensor(new long[2] { 1, expertNum }, 0, deviceId, name: $"{name}.{nameof(m_RouterBias)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);

        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            throw new NotImplementedException();
        }

        public int GetDeviceId()
        {
            throw new NotImplementedException();
        }

        public void ClearStatus()
        {

        }

        public IWeightTensor Process(IWeightTensor input, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null)
        {
            //Computing routing result
            using var g = graph.CreateSubGraph($"{m_name}_MoEFeedForward");
            var inputNorm = layerNorm.Norm(input, g);
            var inputRouterDense = g.Affine(inputNorm, m_Router, m_RouterBias); // [batchSize * seqLen, expertNum]
            var inputRouter = g.Softmax(inputRouterDense); // [batchSize * seqLen, expertNum]


            if (Logger.Verbose == Logger.LogVerbose.Debug)
            {
                var routerArray = inputRouter.ToWeightArray();
                for (int i = 0; i < input.Rows; i++)
                {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < m_expertNum; j++)
                    {
                        sb.Append(routerArray[i * m_expertNum + j]);
                        sb.Append(" ");
                    }

                    Logger.WriteLine($"Token '{i}': '{sb.ToString()}'");
                }

            }

            (var topValue, var topIndex) = g.TopK(inputRouter, m_expertsPerTokenFactor); // [batchSize * seqLen, m_expertsPerTokenFactor]
            var topIndexArray = topIndex.ToWeightArray();
            List<float>[] indexs = new List<float>[m_expertNum]; // [expertNum, token_offsets]
            for (int i = 0; i < indexs.Length; i++)
            {
                indexs[i] = new List<float>();
            }

            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < m_expertsPerTokenFactor; j++)
                {
                    int expertIdx = (int)topIndexArray[i * m_expertsPerTokenFactor + j];
                    indexs[expertIdx].Add(i);
                }
            }

            List<IWeightTensor> tokenEmbsList = new List<IWeightTensor>();
            List<IWeightTensor> tokenIdxList = new List<IWeightTensor>();

            for (int i = 0; i < m_expertNum; i++)
            {
                if (Logger.Verbose == Logger.LogVerbose.Debug)
                {
                    Logger.WriteLine($"Expert '{i}' process '{indexs[i].Count}' tokens.");
                }


                if (indexs[i].Count > 0)
                {
                    using var gExp = g.CreateSubGraph($"{m_name}_MoEFeedForward_{i}");
                    var scores_eI = gExp.AsContiguous(gExp.Peek(inputRouter, 1, i)); // [batchSize * seqLen, 1]
                    var tokenIdx_eI = g.CreateTensorWeights(new long[] { indexs[i].Count, 1 }, indexs[i].ToArray());

                    var topValue_eI = gExp.IndexSelect(scores_eI, tokenIdx_eI); // [indexs[i].Count, 1]
                    topValue_eI = gExp.Expand(topValue_eI, dims: new long[] { indexs[i].Count, inputNorm.Sizes[^1] });

                    var tokenEmbs = gExp.IndexSelect(inputNorm, tokenIdx_eI);
                    var m_Whd1_i = gExp.Select(m_Whd1, 0, i);
                    var m_Whd2_i = gExp.Select(m_Whd2, 0, i);


                    tokenEmbs = gExp.Mul(tokenEmbs, m_Whd1_i);

                    tokenEmbs = RunActivateFunction(gExp, tokenEmbs);

                    tokenEmbs = gExp.Mul(tokenEmbs, m_Whd2_i);
                    tokenEmbs = g.EltMul(tokenEmbs, topValue_eI);

                    tokenEmbsList.Add(tokenEmbs);
                    tokenIdxList.Add(tokenIdx_eI);
                }
            }

            var tokenEmbsAll = g.Concate(tokenEmbsList, 0);
            var tokenIdxAll = g.Concate(tokenIdxList, 0);
            var resultEmbs = g.IndexUpdate(inputNorm.Sizes, tokenEmbsAll, tokenIdxAll, true);

            input = graph.Add(input, resultEmbs);

            return input;
        }

        private IWeightTensor RunActivateFunction(IComputeGraph gExp, IWeightTensor tokenEmbs)
        {
            if (m_activateFunc == ActivateFuncEnums.SiLU)
            {
                tokenEmbs = gExp.SiLU(tokenEmbs, inPlace: true);
            }
            else if (m_activateFunc == ActivateFuncEnums.ReLU)
            {
                tokenEmbs = gExp.ReLU(tokenEmbs, inPlace: true);
            }
            else if (m_activateFunc == ActivateFuncEnums.LeakyReLU)
            {
                tokenEmbs = gExp.LeakyReLU(tokenEmbs, inPlace: true);
            }

            return tokenEmbs;
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(layerNorm.GetParams());

            response.AddRange(m_Whd1.GetParams());
            response.AddRange(m_Whd2.GetParams());

            response.AddRange(m_Router.GetParams());
            response.AddRange(m_RouterBias.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            layerNorm.Save(stream);

            m_Whd1.Save(stream);
            m_Whd2.Save(stream);

            m_Router.Save(stream);
            m_RouterBias.Save(stream);

            stream.AddWeights($"{m_name}.ActivateFunc", new float[1] { (float)m_activateFunc });
        }


        public void Load(IModel stream)
        {
            layerNorm.Load(stream);

            m_Whd1.Load(stream);
            m_Whd2.Load(stream);

            m_Router.Load(stream);
            m_RouterBias.Load(stream);

            m_activateFunc = (ActivateFuncEnums)stream.GetWeights($"{m_name}.ActivateFunc")[0];
            Logger.WriteLine($"Loading '{m_name}' activate function setting '{m_activateFunc}'");

        }
    }
}