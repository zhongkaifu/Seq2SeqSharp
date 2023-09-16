// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        private readonly List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, IComputeGraph graphToBind = null, RandomInitType normType = RandomInitType.None, bool needGradient = true, DType dtype = DType.Float32)
        {
            WeightTensor r = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable, initType: normType, graphToBind: graphToBind, needGradient: needGradient, dtype: dtype);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false, string name = "", IComputeGraph graphToBind = null, RandomInitType normType = RandomInitType.None, bool needGradient = true, DType dtype = DType.Float32)
        {
            WeightTensor r = new WeightTensor(sizes, deviceId, name, initType: normType, graphToBind: graphToBind, needGradient: needGradient, dtype: dtype);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public void Dispose()
        {
            foreach (WeightTensor item in weights)
            {
                item.Dispose();
            }
            weights.Clear();
        }
    }
}
