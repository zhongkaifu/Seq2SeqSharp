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

namespace Seq2SeqSharp.Layers
{
    internal class MoEFeedForward : IFeedForwardLayer
    {
        private readonly LayerNormalization layerNorm;
      //  private readonly LayerNormalization routerNorm;

        private readonly IWeightTensor m_Whd1;
        private readonly IWeightTensor m_Whd2;

        //private readonly IWeightTensor m_ByPassWhd1;
        //private readonly IWeightTensor m_ByPassWhd2;

        private readonly IWeightTensor m_Router;
        private readonly IWeightTensor m_RouterBias;

        private readonly string m_name;
        private readonly int m_expertNum;
        private readonly int m_hiddenDim;
        private readonly int m_expertsPerTokenFactor;

        private ActivateFuncEnums m_activateFunc;

        public MoEFeedForward(string name, int expertNum, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.Relu, int expertsPerTokenFactor = 1)
        {
            m_name = name;
            m_activateFunc = activateFunc;
            m_expertNum = expertNum;
            m_hiddenDim = hiddenDim;
            m_expertsPerTokenFactor = expertsPerTokenFactor;

            Logger.WriteLine($"Creating MoE feed forward layer. Name = '{name}', ExpertNum = '{expertNum}', ExpertsPerToken = '{expertsPerTokenFactor}', HiddenDim = '{hiddenDim}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', Activate Function = '{activateFunc}'");

            layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor);
        //    routerNorm = new LayerNormalization($"{name}.{nameof(routerNorm)}", m_expertNum, deviceId, isTrainable, learningRateFactor: learningRateFactor);

            m_Whd1 = new WeightTensor(new long[3] { expertNum, hiddenDim, hiddenDim * 4 }, deviceId, name: $"{name}.{nameof(m_Whd1)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_Whd2 = new WeightTensor(new long[3] { expertNum, hiddenDim * 4, hiddenDim }, deviceId, name: $"{name}.{nameof(m_Whd2)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);


            //m_ByPassWhd1 = new WeightTensor(new long[3] { expertNum, hiddenDim, hiddenDim * 4 }, deviceId, name: $"{name}.{nameof(m_ByPassWhd1)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            //m_ByPassWhd2 = new WeightTensor(new long[3] { expertNum, hiddenDim * 4, hiddenDim }, deviceId, name: $"{name}.{nameof(m_ByPassWhd2)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);



            m_Router = new WeightTensor(new long[2] { hiddenDim, expertNum}, deviceId, name: $"{name}.{nameof(m_Router)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_RouterBias = new WeightTensor(new long[2] { 1, expertNum }, 0, deviceId, name: $"{name}.{nameof(m_RouterBias)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor);

        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            throw new NotImplementedException();
        }

        public int GetDeviceId()
        {
            throw new NotImplementedException();
        }

        public IWeightTensor Process(IWeightTensor input, int batchSize, IComputeGraph graph)
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

            //###################Token choice top-1 expert###############################
            inputRouter = g.AsContiguous(inputRouter); // [batchSize * seqLen, expertNum]
            (var topValue, var topIndex) = g.TopK(inputRouter, m_expertsPerTokenFactor); // [batchSize * seqLen, m_expertsPerTokenFactor]


            if (g.NeedsBackprop)
            {
                var routerLoss = g.Mean(inputRouter, 0); // [1, expertNum]
                var topKScatter = g.Scatter(topIndex, 1, 1, runGradient: false, shape: inputRouter.Sizes); // [batchSize * seqLen, expertNum]
                topKScatter = g.Mean(topKScatter, 0); // [1, expertNum]

                routerLoss = g.EltMul(routerLoss, topKScatter); // [1, expertNum]
                routerLoss = g.Mean(routerLoss, 1); // [1, 1]
                routerLoss = g.Mul(routerLoss, (float)Math.Sqrt(m_expertNum) * 0.01f);
                routerLoss.FillGradient(1.0f);

                //routerLoss = g.Add(routerLoss, (float)m_expertsPerTokenFactor / (float)m_expertNum);
                //routerLoss = g.Mul(routerLoss, 0.01f);
                //routerLoss.CopyWeightsToGradients(routerLoss);
            }









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

            for (int i = 0; i < m_expertNum; i++)
            {
                if (Logger.Verbose == Logger.LogVerbose.Debug)
                {
                    Logger.WriteLine($"Expert '{i}' process '{indexs[i].Count}' tokens.");
                }


                if (indexs[i].Count > 0)
                {
                    var scores_eI = g.Peek(inputRouter, 1, i); // [batchSize * seqLen, 1]
                    var tokenIdx_eI = g.CreateTensorWeights(new long[] { indexs[i].Count, 1 }, indexs[i].ToArray());

                    var topValue_eI = g.IndexSelect(scores_eI, tokenIdx_eI); // [indexs[i].Count, 1]
                    topValue_eI = g.Expand(topValue_eI, dims: new long[] { indexs[i].Count, inputNorm.Sizes[^1] });

                    var tokenEmbs = g.IndexSelect(inputNorm, tokenIdx_eI);
                    var m_Whd1_i = g.Select(m_Whd1, 0, i);
                    var m_Whd2_i = g.Select(m_Whd2, 0, i);


                    tokenEmbs = g.Mul(tokenEmbs, m_Whd1_i);
                    tokenEmbs = ((m_activateFunc == ActivateFuncEnums.Swish) ? g.Swish(tokenEmbs, inPlace: true) : g.Relu(tokenEmbs, inPlace: true));
                    tokenEmbs = g.Mul(tokenEmbs, m_Whd2_i);
                    tokenEmbs = g.EltMul(tokenEmbs, topValue_eI);

                    var resultEmbs = g.IndexUpdate(inputNorm.Sizes, tokenEmbs, tokenIdx_eI, true);
                    input = g.Add(input, resultEmbs);

                }
            }

            input.UnbindFromComputeGraph();

            return input;

            //////######################End of token choice top-1 expert#####################################









            ////############################Expert choice tokens###################################

            //inputRouter = g.AsContiguous(g.Transpose(inputRouter)); // [expertNum, batchSize * seqLen]

            //int K = (int)Math.Min(input.Rows, input.Rows * m_expertsPerTokenFactor / m_expertNum + 1);
            //(var topKValue, var topKIndex) = g.TopK(inputRouter, K); // [expertNum, K]




            //DumpRoutingResultsInDebugMode(input, K, topKIndex, topKValue);






            //////Collect those tokens that no expert choice
            ////var topKScatter = g.Scatter(topKIndex, 1.0f, 1, runGradient: false, shape: inputRouter.Sizes); // [expertNum, batchSize * seqLen]
            ////topKScatter = g.Sum(topKScatter, 0); // [1, batchSize * seqLen]
            ////var notSelectedTokens = g.EqualTo(topKScatter, 0.0f); // [1, batchSize * seqLen]
            ////int K2 = (int)g.Sum(notSelectedTokens, 1).GetWeightAt(new long[] { 0, 0 });



            //topKIndex = g.AsContiguous(g.View(topKIndex, dims: new long[] { m_expertNum * K, 1 }));
            //topKIndex.UnbindFromComputeGraph();

            //var selectedEmbs = g.IndexSelect(inputNorm, topKIndex, clearWeights: true); // [expertNum * K, hiddenDim]
            //selectedEmbs = g.View(selectedEmbs, dims: new long[] { m_expertNum, K, -1 }); // [expertNum, K, hiddenDim];
            //selectedEmbs = g.MulBatch(selectedEmbs, m_Whd1); // [expertNum, K, hiddenDim * 4]
            //selectedEmbs = ((m_activateFunc == ActivateFuncEnums.Swish) ? g.Swish(selectedEmbs, inPlace: true) : g.Relu(selectedEmbs, inPlace: true));
            //selectedEmbs = g.MulBatch(selectedEmbs, m_Whd2); // [expertNum, K, hiddenDim]

            //topKValue = g.View(topKValue, dims: new long[] { m_expertNum, K, 1 });
            //topKValue = g.Expand(topKValue, dims: new long[] { m_expertNum, K, m_hiddenDim });
            //selectedEmbs = g.EltMul(selectedEmbs, topKValue); // [expertNum, K, hiddenDim]
            //selectedEmbs = g.AsContiguous(g.View(selectedEmbs, dims: new long[] { m_expertNum * K, m_hiddenDim }));

            //var outputEmbs = g.IndexUpdate(input.Sizes, selectedEmbs, topKIndex, true); // [batchSize * seqLen, hiddenDim]
            //outputEmbs = graph.Add(outputEmbs, input);




            ////if (K2 > 0)
            ////{
            ////    notSelectedTokens = g.Expand(notSelectedTokens, dims: new long[] { m_expertNum, input.Rows }); // [expertNum, batchSize * seqLen]
            ////    notSelectedTokens = g.EltMul(inputRouter, notSelectedTokens); // [expertNum, batch * seqLen]

            ////    (var topKValueNotSelect, var topKIndexNotSelect) = g.TopK(notSelectedTokens, K2);

            ////    topKIndexNotSelect = g.AsContiguous(g.View(topKIndexNotSelect, dims: new long[] { m_expertNum * K2, 1 }));
            ////    topKIndexNotSelect.UnbindFromComputeGraph();

            ////    var notSelectedEmbs = g.IndexSelect(inputNorm, topKIndexNotSelect, clearWeights: true); // [expertNum * K2, hiddenDim]
            ////    notSelectedEmbs = 

            ////}









            //return outputEmbs;
        }

        private void DumpRoutingResultsInDebugMode(IWeightTensor input, int K, IWeightTensor topKIndex, IWeightTensor topKValue)
        {
            if (Logger.Verbose == Logger.LogVerbose.Debug)
            {
                Logger.WriteLine($"Input Row = '{input.Rows}', expert size = '{m_expertNum}', K = '{K}', experts per token factor = '{m_expertsPerTokenFactor}'");
                var idxs = topKIndex.ToWeightArray();
                var vals = topKValue.ToWeightArray();
                SortedDictionary<int, int> id2freq = new SortedDictionary<int, int>();
                for (int i = 0; i < m_expertNum; i++)
                {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < K; j++)
                    {
                        int value = (int)idxs[i * K + j];
                        float weight = vals[i * K + j];
                        sb.Append($"{value} ({weight})");
                        sb.Append(" ");

                        if (id2freq.ContainsKey(value) == false)
                        {
                            id2freq.Add(value, 0);
                        }
                        id2freq[value]++;
                    }
                    Logger.WriteLine($"Expert '{i}' has tokens: {sb.ToString()}");
                }

                Logger.WriteLine("Experts size per token:");
                foreach (var pair in id2freq)
                {
                    Logger.WriteLine($"{pair.Key} : {pair.Value}");
                }

                for (int i = 0; i < input.Rows; i++)
                {
                    if (id2freq.ContainsKey(i) == false)
                    {
                        Logger.WriteLine($"token '{i}' is not assigned to any expert.");
                    }
                }
            }
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(layerNorm.GetParams());
         //   response.AddRange(routerNorm.GetParams());

            response.AddRange(m_Whd1.GetParams());
            response.AddRange(m_Whd2.GetParams());


            //response.AddRange(m_ByPassWhd1.GetParams());
            //response.AddRange(m_ByPassWhd2.GetParams());

            response.AddRange(m_Router.GetParams());
            response.AddRange(m_RouterBias.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            layerNorm.Save(stream);
        //    routerNorm.Save(stream);

            m_Whd1.Save(stream);
            m_Whd2.Save(stream);

            //m_ByPassWhd1.Save(stream);
            //m_ByPassWhd2.Save(stream);


            m_Router.Save(stream);
            m_RouterBias.Save(stream);

            stream.AddWeights($"{m_name}.ActivateFunc", new float[1] { (float)m_activateFunc });
        }


        public void Load(IModel stream)
        {
            layerNorm.Load(stream);
         //   routerNorm.Load(stream);

            m_Whd1.Load(stream);
            m_Whd2.Load(stream);

            //m_ByPassWhd1.Load(stream);
            //m_ByPassWhd2.Load(stream);


            m_Router.Load(stream);
            m_RouterBias.Load(stream);

            m_activateFunc = (ActivateFuncEnums)stream.GetWeights($"{m_name}.ActivateFunc")[0];
            Logger.WriteLine($"Loading '{m_name}' activate function setting '{m_activateFunc}'");

        }


    }
}
