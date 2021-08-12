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

        int DeviceId { get; set; }

        float LearningRateFactor { get; set; }

        float GetWeightAt(long[] indices);
        void SetWeightAt(float val, long[] indices);
        void SetGradientAt(float val, long[] indices);

        void CopyWeightsToGradients(IWeightTensor src);

        void SetWeightAtRow(int row, float[] val);

        List<int> GetTopNMaxWeightIdx(int topN);

        void SetWeightArray(float[] v);

        void ReleaseWeight();
        void ReleaseGradient();

        void ZeroGradient();
        void CleanWeight();

        void CopyWeightsFrom(IWeightTensor src);
        void AddGradientFrom(IWeightTensor src);

        float[] ToWeightArray();

        void UnbindFromComputeGraph();

        bool IsWeightNull();
        bool IsGradientNull();

        IAllocator Allocator { get; }


    }
}
