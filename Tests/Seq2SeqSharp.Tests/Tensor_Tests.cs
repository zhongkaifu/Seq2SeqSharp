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
using System.Linq;
using TensorSharp;

namespace Seq2SeqSharp.Tests
{
    [TestClass]
    public class Tensor_Tests
    {
        static Random rnd = new Random(DateTime.Now.Millisecond);

        private Tensor BuildTensor(long[] shape, int deviceId, float value)
        {
            Tensor tensorA = new Tensor(TensorAllocator.Allocator(deviceId), DType.Float32, sizes: shape);

            //Build test data and ground truth data
            float[] arrayA = new float[tensorA.ElementCount()];
            for (int i = 0; i < tensorA.ElementCount(); i++)
            {
                arrayA[i] = value;
            }
            tensorA.SetElementsAsFloat(arrayA);

            return tensorA;
        }

        [TestMethod]
        public void TestScatter()
        {
            int deviceId = 0;
            int N = 100;
            int categoryNum = 10;
            TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { deviceId });
            var tensorIdx = BuildTensor(shape: new long[] { N, 1}, deviceId, 0.0f);
            float[] arrayIdx = new float[N];

            //Build ground truth labels
            for (int i = 0; i < N; i++)
            {
                arrayIdx[i] = rnd.Next(0, categoryNum);
            }
            tensorIdx.SetElementsAsFloat(arrayIdx);

            var resultTensor = BuildTensor(shape: new long[] { N, categoryNum }, deviceId, 0.0f);
            Ops.ScatterFill(resultTensor, 1.0f, 1, tensorIdx);

            float[] arrayResult = new float[N * categoryNum];
            resultTensor.CopyToArray(arrayResult);

            for (int i = 0; i < N; i++)
            {
                Assert.IsTrue(arrayResult[i * categoryNum + (int)arrayIdx[i]] == 1.0f);
                for (int j = 0; j < categoryNum; j++)
                {
                    if (j != (int)arrayIdx[i])
                    {
                        Assert.IsTrue(arrayResult[i * categoryNum + j] == 0.0f);
                    }
                }
            }
        }

        [TestMethod]
        public void TestAtomicAdd()
        {
            int deviceId = 0;
            int N = 456;
            TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { deviceId });

            var tensorA = BuildTensor(shape: new long[] { 100, 200 }, deviceId, 1.23f);
            var tensorAView = tensorA.View(new long[] { 100, 1, 200 });
            var tensorAExp = tensorAView.Expand(new long[] { 100, N, 200 });

            var tensorB = BuildTensor(shape: new long[] { 100, N, 200 }, deviceId, 2.34f);



            Ops.AtomicAdd(tensorAExp, tensorB);


            float[] arr = new float[100 * 200];
            tensorA.CopyToArray(arr);

            float result = (float)Math.Round(1.23f + 2.34f * N, 2);
            for (int i = 0; i < arr.Length; i++)
            {
                float v = (float) Math.Round(arr[i], 2);
                Assert.IsTrue(v == result);
            }
        }

    }
}
