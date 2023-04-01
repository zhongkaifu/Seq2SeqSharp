// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IComputeGraph : IDisposable
    {
        int DeviceId { get; }
        bool NeedsBackprop { get; }
        IComputeGraph CreateSubGraph(string name);
        IWeightTensor Transpose(IWeightTensor w, int dim1, int dim2);
        IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, float alpha = 1.0f);
        IWeightTensor Mul(IWeightTensor w1, IWeightTensor w2, float alpha = 1.0f);
        IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Add(IWeightTensor w1, IWeightTensor w2, bool inPlace = false);
        IWeightTensor Add(IWeightTensor w1, float v);
        IWeightTensor Tanh(IWeightTensor w);
        IWeightTensor Sigmoid(IWeightTensor w);
        IWeightTensor Relu(IWeightTensor w, bool inPlace = false);
        IWeightTensor Swish(IWeightTensor w, bool inPlace = false);

        IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias, float alpha = 1.0f);
        IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4);
        IWeightTensor TransposeBatch(IWeightTensor m, int batchSize);
        IWeightTensor AsContiguous(IWeightTensor w, bool shareTensor = true);
        IWeightTensor View(IWeightTensor w, params long[] dims);
        IWeightTensor Expand(IWeightTensor w, params long[] dims);
        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3);
        IWeightTensor Peek(IWeightTensor w, int dim, int ix, int num = 1);
        IWeightTensor Dropout(IWeightTensor V, int batchSize, float drop_prob, bool inPlace = false);
        IWeightTensor Softmax(IWeightTensor w, bool runGradients = true, bool inPlace = false);
        List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes);
        (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2);
        (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3);
        IWeightTensor Concate(List<IWeightTensor> wl, int dim);

        IWeightTensor Concate(int dim, params IWeightTensor[] wl);
        
        IWeightTensor Transpose(IWeightTensor w);
        IWeightTensor Mul(IWeightTensor w, float v, bool inPlace = false);
        IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-9f);

        IWeightTensor Select(IWeightTensor src, int dim, int index);
        void Backward();
        void VisualizeNeuralNetToFile(string neuralNetPicFilePath);
        IWeightFactory GetWeightFactory();

        IWeightTensor Max(IWeightTensor w, int dim);
        IWeightTensor Argmax(IWeightTensor w, int dim);
        IWeightTensor EqualTo(IWeightTensor w, float val);
        IWeightTensor LessOrEqual(IWeightTensor w, float val);
        IWeightTensor GreaterThan(IWeightTensor w, float val);

        IWeightTensor SampleIndicue(IWeightTensor w, List<List<int>> seqs, float topP = 1.0f, List<int> blockedTokens = null);

        IWeightTensor Zero(long[] sizes);
        IWeightTensor CreateTensorWeights(long[] sizes, float[] values);
        IWeightTensor IndexSelect(IWeightTensor s, IWeightTensor indice, bool clearWeights = false, bool isAdd = false);
        IWeightTensor IndexUpdate(long[] sizes, IWeightTensor s, IWeightTensor indice, bool clearWeights = false);

        void Bind(IWeightTensor w);
        void Unbind(IWeightTensor w);


        IWeightTensor Gather(IWeightTensor src, IWeightTensor indices, int dim, bool runGradients = true);
        IWeightTensor Scatter(IWeightTensor source, IWeightTensor indices, int dim, params long[] shape);
        IWeightTensor Scatter(IWeightTensor indices, float val, int dim, bool runGradient = true, params long[] shape);
        IWeightTensor ScatterAdd(IWeightTensor source, IWeightTensor indices, int dim, params long[] shape);

        (IWeightTensor, IWeightTensor) TopK(IWeightTensor src, int k);
        IWeightTensor Sub(IWeightTensor w0, IWeightTensor w1);
        IWeightTensor Sub(float v, IWeightTensor w1);

        #region Operations for masking
        IWeightTensor BuildSrcTgtMask(int srcPaddedLength, int tgtPaddedLength, float[] tgtOriginalLengths, float[] srcOriginalLengths);
        IWeightTensor BuildSelfTriMask(int paddedLength, float[] originalLengths);
        IWeightTensor BuildTriMask(int paddedLength, int batchSize);
        IWeightTensor BuildPadSelfMask(int paddedLength, float[] originalLengths);
        #endregion

        IWeightTensor LeftShiftTokens(List<List<int>> input, int lastTokenToPad);
        IWeightTensor CreateTokensTensor(List<List<int>> input);

        IWeightTensor BuildFeatureMask(int paddedLength, List<int> appliedLengths, int dim);

        IWeightTensor Sum(IWeightTensor w, int dim);
        IWeightTensor Mean(IWeightTensor w, int dim);
        IWeightTensor Log(IWeightTensor w);

        IWeightTensor Rsqrt(IWeightTensor w);

        IWeightTensor Div(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Div(IWeightTensor w, float v, bool inPlace = false);
        IWeightTensor Exp(IWeightTensor w);
        IWeightTensor Pow(IWeightTensor w, float n);

        float CrossEntropyLoss(IWeightTensor probs, IWeightTensor truthTgtSeqs, float graident = 1.0f, float smooth = 0.0f, float gamma = 0.0f);
        float CrossEntropyLoss(IWeightTensor probs, IWeightTensor truthTgtSeqs, IWeightTensor graident, float smooth = 0.0f, float gamma = 0.0f);
        float NLLLoss(IWeightTensor probs, IWeightTensor truthTgtSeqs, float graident = 1.0f, float smooth = 0.0f);

        IWeightTensor CreateUniformRandomTensor(long[] sizes, float minVal, float maxVal);

        IWeightTensor LogSoftmax(IWeightTensor x);
    }
}
