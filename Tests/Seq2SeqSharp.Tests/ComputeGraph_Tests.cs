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
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;

namespace Seq2SeqSharp.Tests;

[TestClass]
public class ComputeGraph_Tests
{
    static Random rnd = new Random(DateTime.Now.Millisecond);

    private WeightTensor BuildRandomTensor(long[] shape, string name, bool isTrainable = true)
    {
        var tensorA = new WeightTensor(shape, 1, 0, name: name, isTrainable: isTrainable);

        //Build test data and ground truth data
        float[] arrayA = new float[tensorA.ElementCount];
        for (int i = 0; i < tensorA.ElementCount; i++)
        {
            arrayA[i] = (float)rnd.NextDouble();
        }
        tensorA.SetWeightArray(arrayA);

        return tensorA;
    }

    private WeightTensor BuildRandomLabelTensor(int batchSize, int categoryNum, string name)
    {
        var tensorIdx = new WeightTensor(new long[] {batchSize, 1 }, 1, 0, name: name, isTrainable: false);

        //Build ground truth labels
        float[] arrayIdx = new float[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            arrayIdx[i] = rnd.Next(0, categoryNum);
        }
        tensorIdx.SetWeightArray(arrayIdx);

        return tensorIdx;
    }

    [TestMethod]
    public void TestAddTensorTensor()
    {
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = new WeightTensor(new long[2] { 2, 2 }, 1, 0, name: "tensorA", isTrainable: true);
        var tensorB = new WeightTensor(new long[2] { 2, 2 }, 2, 0, name: "tensorB", isTrainable: true);

        var tensorSum = graph.Add(tensorA, tensorB);

        float v = tensorSum.GetWeightAt(new long[] { 1, 1 });

        Assert.IsTrue(v == 3.0f);

    }

    [TestMethod]
    public void TestAddTensorTensorBP()
    {
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = new WeightTensor(new long[2] { 2, 2 }, 1, 0, name: "tensorA", isTrainable: true);
        var tensorB = new WeightTensor(new long[2] { 2, 2 }, 2, 0, name: "tensorB", isTrainable: true);

        var tensorSum = graph.Add(tensorA, tensorB);
        tensorSum.CopyWeightsToGradients(tensorSum);

        graph.Backward();


        float gA = tensorA.GetGradientAt(new long[] { 1, 1 });
        float gB = tensorB.GetGradientAt(new long[] { 1, 1, });

        Assert.IsTrue(gA == 3.0f);
        Assert.IsTrue(gB == 3.0f);
    }

    [TestMethod]
    public void TestAddSubGradients()
    {
        int batchSize = 5;
        int vocabSize = 20;

        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = new WeightTensor(new long[2] { batchSize, vocabSize }, 1, 0, name: "tensorA", isTrainable: true);
        var tensorB = new WeightTensor(new long[2] { batchSize, vocabSize }, 1, 0, name: "tensorB", isTrainable: true);
        var tensorIdx = BuildRandomLabelTensor(batchSize, vocabSize, "tensorIdx");

        var tensorANeg = graph.Mul(tensorA, -1.0f);
        var tensorANegSum = graph.Add(tensorANeg, 100.0f);
        var tensorSub = graph.Sub(100.0f, tensorB);

        float v1 = tensorANegSum.GetWeightAt(new long[] { 1, 1 });
        float v2 = tensorSub.GetWeightAt(new long[] { 1, 1 });

        Assert.IsTrue(v1 == v2);

        var softmax1 = graph.Softmax(tensorANegSum);
        var softmax2 = graph.Softmax(tensorSub);

        graph.CrossEntropyLoss(softmax1, tensorIdx);
        graph.CrossEntropyLoss(softmax2, tensorIdx);

        graph.Backward();

        float gA = tensorA.GetGradientAt(new long[] { 1, 1 });
        float gB = tensorB.GetGradientAt(new long[] { 1, 1, });

        Assert.IsTrue(gA == gB);
    }


    [TestMethod]
    public void TestSum()
    {
        int batchSize = 5;
        int vocabSize = 20;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorA", isTrainable: true);
        var tensorAWeights = tensorA.ToWeightArray();
        float sum1 = tensorAWeights.Sum();

        var tensorSum = graph.Sum(tensorA, 1);
        var tensorSumWeights = tensorSum.ToWeightArray();
        float sum2 = tensorSumWeights.Sum();


        sum1 = (float)Math.Round(sum1, 4);
        sum2 = (float)Math.Round(sum2, 4);

        Logger.WriteLine($"sum from .net core = '{sum1}', sum from sum operator = '{sum2}'");

        Assert.IsTrue(sum1 == sum2);

    }


    [TestMethod]
    public void TestMean()
    {
        int batchSize = 5;
        int vocabSize = 20;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorA", isTrainable: true);
        var tensorMean = graph.Mean(tensorA, 1);

        var tensorWeights = tensorA.ToWeightArray();
        var tensorMeanWegiths = tensorMean.ToWeightArray();


        for (int i = 0; i < batchSize; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < vocabSize; j++)
            {
                sum += tensorWeights[i * vocabSize + j];
            }
            float mean = sum / vocabSize;

            float mean1 = (float)Math.Round(mean, 4);
            float mean2 = (float)Math.Round(tensorMeanWegiths[i], 4);

            Logger.WriteLine($"row '{i}': mean from .net core = '{mean1}', mean from mean operator = '{mean2}'");

            Assert.IsTrue(mean1 == mean2);
        }
    }


    [TestMethod]
    public void TestAtomicAdd()
    {
        int batchSize = 5;
        int vocabSize = 20;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorA", isTrainable: true);
        var tensorAWeights = tensorA.ToWeightArray();
        float sumA = tensorAWeights.Sum();

        var tensorB = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorB", isTrainable: true);
        var tensorBWeights = tensorB.ToWeightArray();
        float sumB = tensorBWeights.Sum();

        float sum = sumA + sumB;

        Ops.AtomicAdd(tensorA.TWeight, tensorB.TWeight);
        var tensorSumWeights = tensorA.ToWeightArray();
        float sum2 = tensorSumWeights.Sum();

        double r1 = Math.Round(sum, 4);
        double r2 = Math.Round(sum2, 4);


        Logger.WriteLine($"sum = '{sum}', sum2 = '{sum2}', r1 = '{r1}', r2 = '{r2}'");

        Assert.IsTrue(r1 == r2);
    }

    [TestMethod]
    public void TestSigmoid()
    {
        int batchSize = 5;
        int vocabSize = 20;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var tensorSrc = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorSrc", isTrainable: true);
        var tensorSrcWeights = tensorSrc.ToWeightArray();

        var tensorRst = new WeightTensor(new long[2] { batchSize, vocabSize }, 1, 0, name: "tensorRst", isTrainable: true);

        Ops.Sigmoid(tensorRst.TWeight, tensorSrc.TWeight);

        var tensorTgtWeights = tensorRst.ToWeightArray();

        for (int i = 0; i < tensorSrcWeights.Length; i++)
        {
            float r1 = (float)(1.0 / (1.0 + Math.Exp(-tensorSrcWeights[i])));
            r1 = (float)Math.Round(r1, 4);

            float r2 = tensorTgtWeights[i];
            r2 = (float)Math.Round(r2, 4);

            Assert.IsTrue(r1 == r2);
        }
    }

    [TestMethod]
    public void TestTopK()
    {
        int batchSize = 256;
        int vocabSize = 25600;
        int K = 1280;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorSrc = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorSrc", isTrainable: true);
        (var tensorK, var tensorIdx) = graph.TopK(tensorSrc, K);

        float[] weightSrc = tensorSrc.ToWeightArray();
        float[] weightK = tensorK.ToWeightArray();
        float[] weightIdx = tensorIdx.ToWeightArray();

        for (int i = 0; i < batchSize; i++)
        {
            SortedDictionary<float, List<int>> sd = new SortedDictionary<float, List<int>>();
            for (int j = 0; j < vocabSize; j++)
            {
                if (sd.ContainsKey(weightSrc[i * vocabSize + j]) == false)
                {
                    sd.Add(weightSrc[i * vocabSize + j], new List<int>());
                }

                sd[weightSrc[i * vocabSize + j]].Add(j);
            }

            List<float> sortedWeights = new List<float>();
            List<float> sortedIdx = new List<float>();
            int cnt = 0;
            foreach (var pair in sd.Reverse())
            {
                if (cnt == K)
                {
                    break;
                }

                foreach (var idx in pair.Value)
                {
                    if (cnt == K)
                    {
                        break;
                    }

                    sortedWeights.Add((float)Math.Round(pair.Key, 4));
                    sortedIdx.Add((float)idx);

                    cnt++;
                }
            }

           

            for (int j = 0; j < K; j++)
            {
                float value = (float)Math.Round(weightK[i * K + j], 4);
                float idx = weightIdx[i * K + j];

                int valueIdx = sortedWeights.IndexOf(value);
                int idxIdx = sortedIdx.IndexOf(idx);

                Assert.IsTrue(valueIdx >= 0);
                Assert.IsTrue(idxIdx >= 0);

                sortedWeights.RemoveAt(valueIdx);
                sortedIdx.RemoveAt(idxIdx);
            }

            Assert.IsTrue(sortedWeights.Count == 0);
            Assert.IsTrue(sortedIdx.Count == 0);
        }
       
    }

    [TestMethod]
    public void TestLogSoftmax()
    {
        int batchSize = 5;
        int vocabSize = 20;
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });
        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = BuildRandomTensor(shape: new long[2] { batchSize, vocabSize }, name: "tensorA", isTrainable: true);

        var probs = graph.Softmax(tensorA);
        var logProbs = graph.Log(probs);
        var logSoftmaxProbs = graph.LogSoftmax(tensorA);

        float[] softmaxWeights = logProbs.ToWeightArray();
        float[] logSoftmaxWeights = logSoftmaxProbs.ToWeightArray();



        //Check if graidents are correct
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < vocabSize; j++)
            {
                float softmaxWeight = softmaxWeights[i * vocabSize + j];
                float logSoftmaxWeight = logSoftmaxWeights[i * vocabSize + j];

                Assert.IsTrue(Math.Round(softmaxWeight, 4) == Math.Round(logSoftmaxWeight, 4));
            }
        }
    }  
}