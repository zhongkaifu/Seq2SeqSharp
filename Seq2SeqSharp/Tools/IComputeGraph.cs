using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IComputeGraph : IDisposable
    {
        IComputeGraph CreateSubGraph(string name);
        IWeightTensor Transpose(IWeightTensor w, int dim1, int dim2);
        IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, float alpha = 1.0f);
        IWeightTensor Mul(IWeightTensor w1, IWeightTensor w2, float alpha = 1.0f);
        IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Add(IWeightTensor w1, IWeightTensor w2, bool runGradient1 = true, bool runGradient2 = true, bool inPlace = false);
        IWeightTensor Add(IWeightTensor w1, float v, bool runGradient = true);
        IWeightTensor Tanh(IWeightTensor w);
        IWeightTensor Sigmoid(IWeightTensor w);
        IWeightTensor Relu(IWeightTensor w, bool inPlace = false);
        IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias, float alpha = 1.0f);
        IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4);
        IWeightTensor TransposeBatch(IWeightTensor m, int batchSize);
        IWeightTensor AsContiguous(IWeightTensor w, bool runGradient = true, bool shareTensor = true);
        IWeightTensor View(IWeightTensor w, bool runGradient = true, params long[] dims);
        IWeightTensor Expand(IWeightTensor w, bool runGradient = true, params long[] dims);
        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3);
        IWeightTensor Peek(IWeightTensor w, int dim, int ix, int num = 1, bool runGradients = true);
        IWeightTensor Dropout(IWeightTensor V, int batchSize, float drop_prob, bool inPlace = false);
        IWeightTensor Softmax(IWeightTensor w, bool runGradients = true, bool inPlace = false);
        IWeightTensor ConcatColumns(params IWeightTensor[] wl);
        List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes);
        (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2);
        (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3);
        IWeightTensor ConcatRows(List<IWeightTensor> wl);
        IWeightTensor Transpose(IWeightTensor w);
        IWeightTensor Mul(IWeightTensor w, float v, bool inPlace = false, bool runGradient = true);
        IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-9f);
        IWeightTensor AddLayerNorm(IWeightTensor src1, IWeightTensor src2, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f);

        IWeightTensor Select(IWeightTensor src, int dim, int index);
        void Backward();
        void VisualizeNeuralNetToFile(string neuralNetPicFilePath);
        IWeightFactory GetWeightFactory();

        IWeightTensor Max(IWeightTensor w, int dim);
        IWeightTensor Argmax(IWeightTensor w, int dim);

        IWeightTensor TopPSampleIndice(IWeightTensor w, List<List<int>> seqs, float topP = 0.9f, float repeatPenalty = 2.0f, float distancePenalty = 10.0f);


        IWeightTensor IndexSelect(IWeightTensor s, float[] idxs);

        void Bind(IWeightTensor w);
        void Unbind(IWeightTensor w);


        IWeightTensor Gather(IWeightTensor src, IWeightTensor indices, int dim);
        IWeightTensor Scatter(IWeightTensor source, IWeightTensor indices, int dim, bool runGradient = true, params long[] shape);
        IWeightTensor Sub(float v, IWeightTensor w1, bool runGradient = true);

        #region Operations for masking
        IWeightTensor BuildSrcTgtMask(int srcPaddedLength, int tgtPaddedLength, float[] tgtOriginalLengths, float[] srcOriginalLengths);
        IWeightTensor BuildSelfTriMask(int paddedLength, float[] originalLengths);
        IWeightTensor BuildTriMask(int paddedLength, int batchSize);
        IWeightTensor BuildPadSelfMask(int paddedLength, float[] originalLengths);
        #endregion

        IWeightTensor LeftShiftTokens(List<List<int>> input, int lastTokenToPad);

        IWeightTensor BuildFeatureMask(int paddedLength, List<int> appliedLengths, int dim);

        IWeightTensor Sum(IWeightTensor w, int dim, bool runGradient = true);
        IWeightTensor Log(IWeightTensor w);

        IWeightTensor Rsqrt(IWeightTensor w);

    }
}
