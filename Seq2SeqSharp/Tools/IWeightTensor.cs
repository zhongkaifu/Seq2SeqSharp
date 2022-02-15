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

        WeightTensor CopyWeightsRef(string name, bool needGradient);

        void CopyWeightsFrom(IWeightTensor src);
        void AddGradientFrom(IWeightTensor src);

        float[] ToWeightArray();

        void UnbindFromComputeGraph();

        bool IsGradientNull();

        IAllocator Allocator { get; }

        void FillGradient(float val);

        void Clamp(float min, float max);

        long ElementCount { get; }

    }
}
