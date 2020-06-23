using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IComputeGraph : IDisposable
    {
        IComputeGraph CreateSubGraph(string name);

        IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, int batchSize, float alpha = 1.0f);
        IWeightTensor Mul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Add(IWeightTensor w1, IWeightTensor w2, bool runGradient1 = true, bool runGradient2 = true);
        IWeightTensor Tanh(IWeightTensor w);
        IWeightTensor Sigmoid(IWeightTensor w);
        IWeightTensor Relu(IWeightTensor w, bool inPlace = false);
        IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias, float alpha = 1.0f);
        IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4);
        IWeightTensor TransposeBatch(IWeightTensor m, int batchSize);
        IWeightTensor Permute(IWeightTensor w, params int[] dims);
        IWeightTensor View(IWeightTensor w, bool runGradient = true, params long[] dims);
        IWeightTensor Expand(IWeightTensor w, bool runGradient = true, params long[] dims);
        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2);

        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3);
        IWeightTensor PeekRow(IWeightTensor w, int ix, int num = 1, bool runGradients = true);
        IWeightTensor Dropout(IWeightTensor V, int batchSize, float drop_prob, bool inPlace = false);
        IWeightTensor Softmax(IWeightTensor w, IWeightTensor mask = null, bool runGradients = true, bool inPlace = false);
        IWeightTensor ConcatColumns(params IWeightTensor[] wl);
        List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes);
        (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2);
        (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3);
        IWeightTensor ConcatRows(List<IWeightTensor> wl);
        IWeightTensor RepeatRows(IWeightTensor w, int n, bool runGradient = true);
        IWeightTensor Transpose(IWeightTensor w);
        IWeightTensor Mul(IWeightTensor w, float v);
        IWeightTensor AddMul(IWeightTensor w1, IWeightTensor w2, float v, bool runGradientW1 = true, bool runGradientW2 = true);
        IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f);
        IWeightTensor AddLayerNorm(IWeightTensor src1, IWeightTensor src2, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f);

        void Backward();
        void RunTopBackward();
        void VisualizeNeuralNetToFile(string neuralNetPicFilePath);
        IWeightFactory GetWeightFactory();

        int[] Argmax(IWeightTensor w, int dim);

        IWeightTensor MaskFill(IWeightTensor w1, IWeightTensor m, float v);

        void Bind(IWeightTensor w);
        void Unbind(IWeightTensor w);

        float UpdateCost(IWeightTensor m, int[] ids);
    }
}
