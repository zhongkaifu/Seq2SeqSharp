// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightTensor : INeuralUnit, IDisposable
    {
        long[] Sizes { get; set; }
        int Rows { get; set; }
        int Columns { get; set; }
        string Name { get; set; }
        bool IsTrainable { get; set; }
        bool NeedGradient { get; set; }
        int DeviceId { get; set; }
        float LearningRateFactor { get; set; }
        DType ElementType { get;}
        IAllocator Allocator { get; }

        float GetWeightAt(long[] indices);
        float GetGradientAt(long[] indices);
        void SetWeightAt(float val, long[] indices);
        void CopyWeightsToGradients(IWeightTensor src);
        List<int> GetTopNMaxWeightIdx(int topN);
        void SetWeightArray(float[] v);
        void ReleaseWeight();
        void ReleaseGradient();
        void ZeroGradient();
        void CleanWeight();
        WeightTensor CopyWeightsRef(string name, bool needGradient, IComputeGraph graphToBind);
        void CopyWeightsFrom(IWeightTensor src);
        void AddGradientFrom(IWeightTensor src);
        float[] ToWeightArray();
        void UnbindFromComputeGraph();
        bool IsGradientNull();
        void FillGradient(float val);
        void Clamp(float min, float max);
        long ElementCount { get; }
        void PrintWeights();
        bool IsWeightsCorrupted();
    }
}
