using System;
using TensorSharp.Core;

namespace TensorSharp.GGML
{
    [OpsClass]
    public class GgmlBasicOps
    {
        [RegisterOpStorageType("fill", typeof(GgmlStorage))]
        public static unsafe void Fill(Tensor result, float value)
        {
            ValidateGgmlTensor(result, nameof(result), "fill");

            float* buffer = (float*)GetBufferStart(result);
            TensorIterState iter = new TensorIterState(buffer, result.DimensionCount, result.Sizes, result.Strides);

            do
            {
                for (; !iter.ReachedBlockEnd(); iter.BlockStep())
                {
                    *iter.data = value;
                }
            } while (iter.NextBlock());
        }

        [RegisterOpStorageType("buildtrimask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildTriMask(Tensor result, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildtrimask");
            GetFlatRowsCols(result, "buildtrimask", out int rows, out int cols);

            float* resultPtr = (float*)GetBufferStart(result);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = col <= row ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildselfmask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSelfMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildselfmask");
            ValidateMaskLengthsTensor(originalLengths, nameof(originalLengths), "buildselfmask");
            GetFlatRowsCols(result, "buildselfmask", out int rows, out int cols);

            if (paddedSeqLen <= 0 || (rows % paddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildselfmask expects rows to be divisible by paddedSeqLen.");
            }

            int batchSize = rows / paddedSeqLen;
            if (originalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildselfmask expects one original length per batch item.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* originalLengthsPtr = (float*)GetBufferStart(originalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / paddedSeqLen;
                int seqIdxInBatch = row % paddedSeqLen;
                int originalLength = (int)originalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < originalLength && seqIdxInBatch < originalLength) ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildselftrimask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSelfTriMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildselftrimask");
            ValidateMaskLengthsTensor(originalLengths, nameof(originalLengths), "buildselftrimask");
            GetFlatRowsCols(result, "buildselftrimask", out int rows, out int cols);

            if (paddedSeqLen <= 0 || (rows % paddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildselftrimask expects rows to be divisible by paddedSeqLen.");
            }

            int batchSize = rows / paddedSeqLen;
            if (originalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildselftrimask expects one original length per batch item.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* originalLengthsPtr = (float*)GetBufferStart(originalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / paddedSeqLen;
                int seqIdxInBatch = row % paddedSeqLen;
                int originalLength = (int)originalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < originalLength && seqIdxInBatch < originalLength && col <= seqIdxInBatch) ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("buildsrctgtmask", typeof(GgmlStorage))]
        public static unsafe Tensor BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLen, int tgtPaddedSeqLen, float value, float maskedValue)
        {
            ValidateMaskResultTensor(result, "buildsrctgtmask");
            ValidateMaskLengthsTensor(srcOriginalLengths, nameof(srcOriginalLengths), "buildsrctgtmask");
            ValidateMaskLengthsTensor(tgtOriginalLengths, nameof(tgtOriginalLengths), "buildsrctgtmask");
            GetFlatRowsCols(result, "buildsrctgtmask", out int rows, out int cols);

            if (tgtPaddedSeqLen <= 0 || (rows % tgtPaddedSeqLen) != 0)
            {
                throw new InvalidOperationException("buildsrctgtmask expects rows to be divisible by tgtPaddedSeqLen.");
            }

            int batchSize = rows / tgtPaddedSeqLen;
            if (srcOriginalLengths.ElementCount() != batchSize || tgtOriginalLengths.ElementCount() != batchSize)
            {
                throw new InvalidOperationException("buildsrctgtmask expects source and target length tensors to match batch size.");
            }

            if (cols != srcPaddedSeqLen)
            {
                throw new InvalidOperationException("buildsrctgtmask expects the result last dimension to equal srcPaddedSeqLen.");
            }

            float* resultPtr = (float*)GetBufferStart(result);
            float* srcOriginalLengthsPtr = (float*)GetBufferStart(srcOriginalLengths);
            float* tgtOriginalLengthsPtr = (float*)GetBufferStart(tgtOriginalLengths);
            for (int row = 0; row < rows; ++row)
            {
                float* resultRow = resultPtr + row * cols;
                int batchIdx = row / tgtPaddedSeqLen;
                int seqIdxInBatch = row % tgtPaddedSeqLen;
                int srcOriginalLength = (int)srcOriginalLengthsPtr[batchIdx];
                int tgtOriginalLength = (int)tgtOriginalLengthsPtr[batchIdx];

                for (int col = 0; col < cols; ++col)
                {
                    resultRow[col] = (col < srcOriginalLength && seqIdxInBatch < tgtOriginalLength) ? value : maskedValue;
                }
            }

            return result;
        }

        [RegisterOpStorageType("gather", typeof(GgmlStorage))]
        public static unsafe Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateGatherArguments(result, src, dim, indices);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);
            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = (long)*(indicesIter.data + i * indicesIter.stride);
                    if (idx < 0 || idx >= srcIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', srcSize = '{srcIter.size}'");
                    }

                    *(resultIter.data + i * resultIter.stride) = *(srcIter.data + idx * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("scatter", typeof(GgmlStorage))]
        public static unsafe Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateScatterArguments(result, src, dim, indices, "scatter");

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = (long)*(indicesIter.data + i * indicesIter.stride);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) = *(srcIter.data + i * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("scatter_add", typeof(GgmlStorage))]
        public static unsafe Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateScatterArguments(result, src, dim, indices, "scatter_add");

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = (long)*(indicesIter.data + i * indicesIter.stride);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter_add. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) += *(srcIter.data + i * srcIter.stride);
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("scatter_fill", typeof(GgmlStorage))]
        public static unsafe Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            ValidateScatterFillArguments(result, dim, indices);

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides, dim);
            TensorDimIterState indicesIter = new TensorDimIterState((float*)GetBufferStart(indices), indices.DimensionCount, indices.Sizes, indices.Strides, dim);

            do
            {
                for (int i = 0; i < indicesIter.size; ++i)
                {
                    long idx = (long)*(indicesIter.data + i * indicesIter.stride);
                    if (idx < 0 || idx >= resultIter.size)
                    {
                        throw new IndexOutOfRangeException($"Invalid index in scatter_fill. Idx = '{idx}', resultSize = '{resultIter.size}'");
                    }

                    *(resultIter.data + idx * resultIter.stride) = value;
                }
            } while (resultIter.NextBlock() && indicesIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("addmm", typeof(GgmlStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            ValidateAddmmArguments(result, src, m1, m2);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView2D srcView = default;
            Tensor compactSrc = null;
            try
            {
                if (!TryCreateStandardView(writeTarget, out GgmlTensorView2D resultView)
                    || (beta != 0.0f && !TryCreateStandardOrBroadcastView(src, out srcView, out compactSrc))
                    || !TryCreateRawView(m1, out GgmlTensorView2D m1View)
                    || !TryCreateRawView(m2, out GgmlTensorView2D m2View))
                {
                    throw new NotSupportedException("GGML addmm requires Float32 tensors with supported row-contiguous/view-compatible layouts.");
                }

                if (beta == 0.0f)
                {
                    srcView = default;
                }

                GgmlNative.Addmm(resultView, srcView, m1View, m2View, beta, alpha);
            }
            finally
            {
                compactSrc?.Dispose();
            }

            return writeTarget;
        }

        [RegisterOpStorageType("addmmbatch", typeof(GgmlStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            ValidateAddmmBatchArguments(result, src, m1, m2);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView3D srcView = default;
            Tensor compactSrc = null;
            try
            {
                if (!TryCreateStandardView(writeTarget, out GgmlTensorView3D resultView)
                    || (beta != 0.0f && !TryCreateStandardOrBroadcastView(src, out srcView, out compactSrc))
                    || !TryCreateRawView(m1, out GgmlTensorView3D m1View)
                    || !TryCreateRawView(m2, out GgmlTensorView3D m2View))
                {
                    throw new NotSupportedException("GGML addmmbatch requires Float32 tensors with supported row-contiguous/view-compatible layouts.");
                }

                if (beta == 0.0f)
                {
                    srcView = default;
                }

                GgmlNative.AddmmBatch(resultView, srcView, m1View, m2View, beta, alpha);
            }
            finally
            {
                compactSrc?.Dispose();
            }

            return writeTarget;
        }

        [RegisterOpStorageType("softmax", typeof(GgmlStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            ValidateSoftmaxArguments(result, src);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException("GGML softmax requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Softmax(resultView, srcView);
            return writeTarget;
        }

        [RegisterOpStorageType("softmaxgrad", typeof(GgmlStorage))]
        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            ValidateSoftmaxGradArguments(grad, adj, val);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad, adj, false, adj.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView)
                || !TryCreateStandardView(val, out GgmlTensorView4D valView))
            {
                throw new NotSupportedException("GGML softmaxgrad requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.SoftmaxGrad(resultView, adjView, valView, addGrad);
            return writeTarget;
        }

        [RegisterOpStorageType("adam", typeof(GgmlStorage))]
        public static Tensor Adam(
            Tensor tw,
            Tensor tg,
            Tensor tv,
            Tensor tm,
            float gradNormFactor,
            float stepSize,
            float clipval,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            ValidateAdamArguments(tw, tg, tv, tm);

            if (!TryCreateContiguousTensor(tw, out GgmlContiguousTensor weight)
                || !TryCreateContiguousTensor(tg, out GgmlContiguousTensor gradient)
                || !TryCreateContiguousTensor(tv, out GgmlContiguousTensor v)
                || !TryCreateContiguousTensor(tm, out GgmlContiguousTensor m))
            {
                throw new NotSupportedException("GGML Adam requires contiguous Float32 tensors.");
            }

            GgmlNative.Adam(weight, gradient, v, m, gradNormFactor, stepSize, clipval, regc, decayRateV, decayRateM, iter, eps);
            return tw;
        }

        [RegisterOpStorageType("copy", typeof(GgmlStorage))]
        public static unsafe void Copy(Tensor result, Tensor src)
        {
            ValidateCopyArguments(result, src);

            long elementCount = result.ElementCount();
            if (elementCount == 0)
            {
                return;
            }

            float* resultBuffer = (float*)GetBufferStart(result);
            float* srcBuffer = (float*)GetBufferStart(src);

            if (result.IsContiguous() && src.IsContiguous())
            {
                long byteCount = checked(elementCount * result.ElementType.Size());
                Buffer.MemoryCopy(srcBuffer, resultBuffer, byteCount, byteCount);
                return;
            }

            TensorIterState resultIter = new TensorIterState(resultBuffer, result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState srcIter = new TensorIterState(srcBuffer, src.DimensionCount, src.Sizes, src.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !srcIter.ReachedBlockEnd(); resultIter.BlockStep(), srcIter.BlockStep())
                {
                    *resultIter.data = *srcIter.data;
                }
            } while (resultIter.NextBlock() && srcIter.NextBlock());
        }

        [RegisterOpStorageType("sum", typeof(GgmlStorage))]
        public static unsafe Tensor Sum(Tensor result, Tensor src, int dimension) => ExecuteReduction(result, src, dimension, false, "sum");

        [RegisterOpStorageType("mean", typeof(GgmlStorage))]
        public static unsafe Tensor Mean(Tensor result, Tensor src, int dimension) => ExecuteReduction(result, src, dimension, true, "mean");

        [RegisterOpStorageType("argmin", typeof(GgmlStorage))]
        public static unsafe Tensor Argmin(Tensor result, Tensor src, int dimension) => ExecuteIndexReduction(result, src, dimension, true, "argmin");

        [RegisterOpStorageType("argmax", typeof(GgmlStorage))]
        public static unsafe Tensor Argmax(Tensor result, Tensor src, int dimension) => ExecuteIndexReduction(result, src, dimension, false, "argmax");

        [RegisterOpStorageType("iscorrupted", typeof(GgmlStorage))]
        public static unsafe bool IsCorrupted(Tensor src)
        {
            ValidateGgmlTensor(src, nameof(src), "iscorrupted");

            float* buffer = (float*)GetBufferStart(src);
            TensorIterState iter = new TensorIterState(buffer, src.DimensionCount, src.Sizes, src.Strides);
            do
            {
                for (; !iter.ReachedBlockEnd(); iter.BlockStep())
                {
                    if (!float.IsFinite(*iter.data))
                    {
                        return true;
                    }
                }
            } while (iter.NextBlock());

            return false;
        }

        [RegisterOpStorageType("abs", typeof(GgmlStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Abs, "abs");

        [RegisterOpStorageType("neg", typeof(GgmlStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Neg, "neg");

        [RegisterOpStorageType("sqrt", typeof(GgmlStorage))]
        public static Tensor Sqrt(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Sqrt, "sqrt");

        [RegisterOpStorageType("exp", typeof(GgmlStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Exp, "exp");

        [RegisterOpStorageType("log", typeof(GgmlStorage))]
        public static Tensor Log(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Log, "log");

        [RegisterOpStorageType("relu", typeof(GgmlStorage))]
        public static Tensor Relu(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Relu, "relu");

        [RegisterOpStorageType("sigmoid", typeof(GgmlStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Sigmoid, "sigmoid");

        [RegisterOpStorageType("tanh", typeof(GgmlStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.Tanh, "tanh");

        [RegisterOpStorageType("SiLU", typeof(GgmlStorage))]
        public static Tensor SiLU(Tensor result, Tensor src) => ExecuteUnary(result, src, GgmlUnaryOp.SiLU, "SiLU");

        [RegisterOpStorageType("addt", typeof(GgmlStorage))]
        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Add, "addt");

        [RegisterOpStorageType("subt", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Sub, "subt");

        [RegisterOpStorageType("mult", typeof(GgmlStorage))]
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Mul, "mult");

        [RegisterOpStorageType("divt", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) => ExecuteBinaryTensor(result, lhs, rhs, GgmlBinaryTensorOp.Div, "divt");

        [RegisterOpStorageType("addv", typeof(GgmlStorage))]
        public static Tensor Add(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Add, "addv");

        [RegisterOpStorageType("subv", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Sub, "subv");

        [RegisterOpStorageType("rsubv", typeof(GgmlStorage))]
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) => ExecuteBinaryScalar(result, rhs, lhs, GgmlBinaryScalarOp.ReverseSub, "rsubv");

        [RegisterOpStorageType("mulv", typeof(GgmlStorage))]
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Mul, "mulv");

        [RegisterOpStorageType("divv", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) => ExecuteBinaryScalar(result, lhs, rhs, GgmlBinaryScalarOp.Div, "divv");

        [RegisterOpStorageType("rdivv", typeof(GgmlStorage))]
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) => ExecuteBinaryScalar(result, rhs, lhs, GgmlBinaryScalarOp.ReverseDiv, "rdivv");

        [RegisterOpStorageType("addmul", typeof(GgmlStorage))]
        public static unsafe Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            ValidateElementwiseArguments(result, "addmul", x, y, z);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data * *zIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("adddiv", typeof(GgmlStorage))]
        public static unsafe Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            ValidateElementwiseArguments(result, "adddiv", x, y, z);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data / *zIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("addmulv", typeof(GgmlStorage))]
        public static unsafe Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            ValidateElementwiseArguments(result, "addmulv", x, y);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep())
                {
                    *resultIter.data = *xIter.data + (*yIter.data * z);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("mulmuladd", typeof(GgmlStorage))]
        public static unsafe Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w)
        {
            ValidateElementwiseArguments(result, "mulmuladd", x, y, z, w);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides);
            TensorIterState xIter = new TensorIterState((float*)GetBufferStart(x), x.DimensionCount, x.Sizes, x.Strides);
            TensorIterState yIter = new TensorIterState((float*)GetBufferStart(y), y.DimensionCount, y.Sizes, y.Strides);
            TensorIterState zIter = new TensorIterState((float*)GetBufferStart(z), z.DimensionCount, z.Sizes, z.Strides);
            TensorIterState wIter = new TensorIterState((float*)GetBufferStart(w), w.DimensionCount, w.Sizes, w.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !xIter.ReachedBlockEnd() && !yIter.ReachedBlockEnd() && !zIter.ReachedBlockEnd() && !wIter.ReachedBlockEnd();
                    resultIter.BlockStep(), xIter.BlockStep(), yIter.BlockStep(), zIter.BlockStep(), wIter.BlockStep())
                {
                    *resultIter.data = (*xIter.data * *yIter.data) + (*zIter.data * *wIter.data);
                }
            } while (resultIter.NextBlock() && xIter.NextBlock() && yIter.NextBlock() && zIter.NextBlock() && wIter.NextBlock());

            return writeTarget;
        }

        [RegisterOpStorageType("relud", typeof(GgmlStorage))]
        public static Tensor ReluD(Tensor result, Tensor w, Tensor g) => ExecuteActivationGrad(result, null, w, g, GgmlActivationGradOp.Relu, "relud");

        [RegisterOpStorageType("addrelud", typeof(GgmlStorage))]
        public static Tensor AddReluD(Tensor result, Tensor src, Tensor w, Tensor g) => ExecuteActivationGrad(result, src, w, g, GgmlActivationGradOp.Relu, "addrelud");

        [RegisterOpStorageType("sigmoidD", typeof(GgmlStorage))]
        public static Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, null, resW, resG, GgmlActivationGradOp.Sigmoid, "sigmoidD");

        [RegisterOpStorageType("addsigmoidD", typeof(GgmlStorage))]
        public static Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, t, resW, resG, GgmlActivationGradOp.Sigmoid, "addsigmoidD");

        [RegisterOpStorageType("tanhD", typeof(GgmlStorage))]
        public static Tensor TanhD(Tensor result, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, null, resW, resG, GgmlActivationGradOp.Tanh, "tanhD");

        [RegisterOpStorageType("addtanhD", typeof(GgmlStorage))]
        public static Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) => ExecuteActivationGrad(result, t, resW, resG, GgmlActivationGradOp.Tanh, "addtanhD");

        [RegisterOpStorageType("SiLUD", typeof(GgmlStorage))]
        public static Tensor SiLUD(Tensor result, Tensor srcW, Tensor resG) => ExecuteActivationGrad(result, null, srcW, resG, GgmlActivationGradOp.SiLU, "SiLUD");

        [RegisterOpStorageType("AddSiLUD", typeof(GgmlStorage))]
        public static Tensor AddSiLUD(Tensor result, Tensor srcG, Tensor srcW, Tensor resG) => ExecuteActivationGrad(result, srcG, srcW, resG, GgmlActivationGradOp.SiLU, "AddSiLUD");

        [RegisterOpStorageType("layernorm", typeof(GgmlStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps)
            => ExecuteNorm(result, src, gamma, beta, eps, GgmlNormOp.LayerNorm, "layernorm");

        [RegisterOpStorageType("rmsnorm", typeof(GgmlStorage))]
        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps)
            => ExecuteNorm(result, src, gamma, beta, eps, GgmlNormOp.RmsNorm, "rmsnorm");

        [RegisterOpStorageType("layernormgrad", typeof(GgmlStorage))]
        public static Tensor LayerNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps)
            => ExecuteNormGrad(result, gradGamma, gradBeta, adj, y, x, gamma, beta, eps, GgmlNormOp.LayerNorm, "layernormgrad");

        [RegisterOpStorageType("rmsnormgrad", typeof(GgmlStorage))]
        public static Tensor RMSNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps)
            => ExecuteNormGrad(result, gradGamma, gradBeta, adj, y, x, gamma, beta, eps, GgmlNormOp.RmsNorm, "rmsnormgrad");

        [RegisterOpStorageType("indexselect", typeof(GgmlStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            ValidateIndexSelectArguments(result, src, indice, isAdd);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, new long[] { indice.Sizes[0], src.Sizes[1] });
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView2D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView2D srcView)
                || !TryCreateContiguousTensor(indice, out GgmlContiguousTensor indexTensor))
            {
                throw new NotSupportedException("GGML indexselect requires Float32 row-contiguous source/result matrices and a contiguous Float32 index vector.");
            }

            GgmlNative.IndexSelect(resultView, srcView, indexTensor, isAdd);
            return writeTarget;
        }

        [RegisterOpStorageType("indexselectgrad", typeof(GgmlStorage))]
        public static Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            ValidateIndexSelectGradArguments(grad, adj, indice);

            if (!TryCreateStandardView(grad, out GgmlTensorView2D gradView)
                || !TryCreateStandardView(adj, out GgmlTensorView2D adjView)
                || !TryCreateContiguousTensor(indice, out GgmlContiguousTensor indexTensor))
            {
                throw new NotSupportedException("GGML indexselectgrad requires Float32 row-contiguous gradient/adjoint matrices and a contiguous Float32 index vector.");
            }

            GgmlNative.IndexSelectGrad(gradView, adjView, indexTensor);
            return grad;
        }

        [RegisterOpStorageType("rope", typeof(GgmlStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            ValidateRoPEArguments(result, src, seqLen, "rope");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException("GGML rope requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.RoPE(resultView, srcView, seqLen, rowOffset);
            return writeTarget;
        }

        [RegisterOpStorageType("ropegrad", typeof(GgmlStorage))]
        public static Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen, int rowOffset)
        {
            ValidateRoPEArguments(grad, adj, seqLen, "ropegrad");

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad, adj, false, adj.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView))
            {
                throw new NotSupportedException("GGML ropegrad requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.RoPEGrad(resultView, adjView, seqLen, rowOffset);
            return writeTarget;
        }

        [RegisterOpStorageType("float2half", typeof(GgmlStorage))]
        public Tensor Float2Half(Tensor result, Tensor src)
        {
            throw new NotSupportedException("The GGML Metal backend currently supports Float32 tensors only. Disable AMP to use this backend.");
        }

        [RegisterOpStorageType("half2float", typeof(GgmlStorage))]
        public Tensor Half2Float(Tensor result, Tensor src)
        {
            throw new NotSupportedException("The GGML Metal backend currently supports Float32 tensors only. Disable AMP to use this backend.");
        }

        private static Tensor ExecuteUnary(Tensor result, Tensor src, GgmlUnaryOp op, string opName)
        {
            ValidateUnaryArguments(result, src, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Unary(op, resultView, srcView);
            return writeTarget;
        }

        private static Tensor ExecuteBinaryTensor(Tensor result, Tensor lhs, Tensor rhs, GgmlBinaryTensorOp op, string opName)
        {
            ValidateBinaryTensorArguments(result, lhs, rhs, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            Tensor compactRhs = null;
            try
            {
                if (TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                    && TryCreateStandardView(lhs, out GgmlTensorView4D lhsView)
                    && TryCreateStandardOrBroadcastView(rhs, out GgmlTensorView4D rhsView, out compactRhs))
                {
                    GgmlNative.BinaryTensor(op, resultView, lhsView, rhsView);
                    return writeTarget;
                }

                if (op == GgmlBinaryTensorOp.Add
                    && HasSameShape(writeTarget, rhs)
                    && AreEquivalentViews(writeTarget, lhs)
                    && HasExpandedWriteDimension(writeTarget))
                {
                    return ExecuteAtomicAddHost(writeTarget, rhs);
                }

                if (!CanUseHostElementwiseWrite(writeTarget))
                {
                    throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous or broadcast-expand compatible layout.");
                }

                return ExecuteBinaryTensorHost(writeTarget, lhs, rhs, op, opName);
            }
            finally
            {
                compactRhs?.Dispose();
            }
        }

        private static Tensor ExecuteBinaryScalar(Tensor result, Tensor src, float scalar, GgmlBinaryScalarOp op, string opName)
        {
            ValidateBinaryScalarArguments(result, src, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.BinaryScalar(op, resultView, srcView, scalar);
            return writeTarget;
        }

        private static unsafe Tensor ExecuteBinaryTensorHost(Tensor result, Tensor lhs, Tensor rhs, GgmlBinaryTensorOp op, string opName)
        {
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState lhsIter = new TensorIterState((float*)GetBufferStart(lhs), lhs.DimensionCount, lhs.Sizes, lhs.Strides);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.Sizes, rhs.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !lhsIter.ReachedBlockEnd() && !rhsIter.ReachedBlockEnd();
                    resultIter.BlockStep(), lhsIter.BlockStep(), rhsIter.BlockStep())
                {
                    float lhsValue = *lhsIter.data;
                    float rhsValue = *rhsIter.data;
                    *resultIter.data = ApplyBinaryTensorOp(lhsValue, rhsValue, op, opName);
                }
            } while (resultIter.NextBlock() && lhsIter.NextBlock() && rhsIter.NextBlock());

            return result;
        }

        [RegisterOpStorageType("atomicadd", typeof(GgmlStorage))]
        public static Tensor AtomicAdd(Tensor result, Tensor rhs)
        {
            ValidateBinaryTensorArguments(result, result, rhs, "atomicadd");
            return ExecuteAtomicAddHost(result, rhs);
        }

        private static unsafe Tensor ExecuteAtomicAddHost(Tensor result, Tensor rhs)
        {
            TensorIterState resultIter = new TensorIterState((float*)GetBufferStart(result), result.DimensionCount, result.Sizes, result.Strides);
            TensorIterState rhsIter = new TensorIterState((float*)GetBufferStart(rhs), rhs.DimensionCount, rhs.Sizes, rhs.Strides);

            do
            {
                for (; !resultIter.ReachedBlockEnd() && !rhsIter.ReachedBlockEnd(); resultIter.BlockStep(), rhsIter.BlockStep())
                {
                    *resultIter.data += *rhsIter.data;
                }
            } while (resultIter.NextBlock() && rhsIter.NextBlock());

            return result;
        }

        private static float ApplyBinaryTensorOp(float lhsValue, float rhsValue, GgmlBinaryTensorOp op, string opName)
        {
            return op switch
            {
                GgmlBinaryTensorOp.Add => lhsValue + rhsValue,
                GgmlBinaryTensorOp.Sub => lhsValue - rhsValue,
                GgmlBinaryTensorOp.Mul => lhsValue * rhsValue,
                GgmlBinaryTensorOp.Div => lhsValue / rhsValue,
                _ => throw new NotSupportedException($"Host fallback does not support GGML binary tensor op '{opName}'."),
            };
        }

        private static Tensor ExecuteActivationGrad(Tensor result, Tensor accumulation, Tensor src, Tensor grad, GgmlActivationGradOp op, string opName)
        {
            ValidateActivationGradArguments(result, accumulation, src, grad, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            GgmlTensorView4D accumulationView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateStandardView(grad, out GgmlTensorView4D gradView)
                || (accumulation != null && !TryCreateStandardView(accumulation, out accumulationView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.ActivationGrad(op, resultView, srcView, gradView, accumulationView, accumulation != null);
            return writeTarget;
        }

        private static Tensor ExecuteNorm(Tensor result, Tensor src, Tensor gamma, Tensor beta, float eps, GgmlNormOp op, string opName)
        {
            ValidateNormArguments(result, src, gamma, beta, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            GgmlTensorView4D betaView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView)
                || !TryCreateStandardView(gamma, out GgmlTensorView4D gammaView)
                || (beta != null && !TryCreateStandardView(beta, out betaView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 1 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.Norm(op, resultView, srcView, gammaView, betaView, beta != null, eps);
            return writeTarget;
        }

        private static Tensor ExecuteNormGrad(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, float eps, GgmlNormOp op, string opName)
        {
            ValidateNormGradArguments(result, gradGamma, gradBeta, adj, y, x, gamma, beta, opName);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, adj, false, adj.Sizes);
            GgmlTensorView4D gradBetaView = default;
            if (!TryCreateStandardView(writeTarget, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(gradGamma, out GgmlTensorView4D gradGammaView)
                || !TryCreateStandardView(adj, out GgmlTensorView4D adjView)
                || !TryCreateStandardView(x, out GgmlTensorView4D xView)
                || !TryCreateStandardView(gamma, out GgmlTensorView4D gammaView)
                || (gradBeta != null && !TryCreateStandardView(gradBeta, out gradBetaView)))
            {
                throw new NotSupportedException($"GGML {opName} requires Float32 tensors with 2 to 4 dimensions and a row-contiguous layout.");
            }

            GgmlNative.NormGrad(op, resultView, gradGammaView, gradBetaView, adjView, xView, gammaView, gradBeta != null, eps);
            return writeTarget;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView2D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor) || !TryCreateRawView(tensor, out GgmlTensorView2D rawView))
            {
                view = default;
                return false;
            }

            view = rawView;
            return true;
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView2D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView3D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor) || !TryCreateRawView(tensor, out GgmlTensorView3D rawView))
            {
                view = default;
                return false;
            }

            view = rawView;
            return true;
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView3D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateStandardView(Tensor tensor, out GgmlTensorView4D view)
        {
            if (!CanMapTensorToStandardGgmlView(tensor)
                || tensor.DimensionCount > 4
                || !(tensor.Storage is GgmlStorage)
                || !TryGetRawSpanBytes(tensor, out long rawBytes))
            {
                view = default;
                return false;
            }

            long[] paddedSizes = new long[] { 1, 1, 1, 1 };
            long[] paddedStrides = new long[] { 0, 0, 0, 0 };
            int offset = 4 - tensor.DimensionCount;

            for (int i = 0; i < tensor.DimensionCount; ++i)
            {
                paddedSizes[offset + i] = tensor.Sizes[i];
                paddedStrides[offset + i] = tensor.Strides[i];
            }

            try
            {
                for (int i = 2; i >= 0; --i)
                {
                    long requiredStride = checked(paddedStrides[i + 1] * paddedSizes[i + 1]);
                    if (paddedSizes[i] == 1 || i < offset)
                    {
                        paddedStrides[i] = requiredStride;
                    }
                }

                if (!TryGetInt32(paddedSizes[3], out int ne0)
                    || !TryGetInt32(paddedSizes[2], out int ne1)
                    || !TryGetInt32(paddedSizes[1], out int ne2)
                    || !TryGetInt32(paddedSizes[0], out int ne3))
                {
                    view = default;
                    return false;
                }

                long elementSize = tensor.ElementType.Size();
                long nb1 = checked(paddedStrides[2] * elementSize);
                long nb2 = checked(paddedStrides[1] * elementSize);
                long nb3 = checked(paddedStrides[0] * elementSize);

                view = new GgmlTensorView4D(GetBufferStart(tensor), ne0, ne1, ne2, ne3, nb1, nb2, nb3, rawBytes);
                return true;
            }
            catch (OverflowException)
            {
                view = default;
                return false;
            }
        }

        private static bool TryCreateStandardOrBroadcastView(Tensor tensor, out GgmlTensorView4D view, out Tensor compactTensor)
        {
            if (TryCreateStandardView(tensor, out view))
            {
                compactTensor = null;
                return true;
            }

            if (!TryCreateCompactedBroadcastTensor(tensor, out compactTensor) || !TryCreateStandardView(compactTensor, out view))
            {
                compactTensor?.Dispose();
                compactTensor = null;
                view = default;
                return false;
            }

            return true;
        }

        private static bool TryCreateContiguousTensor(Tensor tensor, out GgmlContiguousTensor contiguousTensor)
        {
            if (tensor.ElementType != DType.Float32 || !(tensor.Storage is GgmlStorage) || !tensor.IsContiguous())
            {
                contiguousTensor = default;
                return false;
            }

            contiguousTensor = new GgmlContiguousTensor(GetBufferStart(tensor), tensor.ElementCount());
            return true;
        }

        private static bool TryCreateRawView(Tensor tensor, out GgmlTensorView2D view)
        {
            if (tensor.DimensionCount != 2
                || tensor.ElementType != DType.Float32
                || !(tensor.Storage is GgmlStorage)
                || !TryGetRawSpanBytes(tensor, out long rawBytes)
                || !TryGetInt32(tensor.Sizes[0], out int dim0)
                || !TryGetInt32(tensor.Sizes[1], out int dim1)
                || !TryGetInt32(tensor.Strides[0], out int stride0)
                || !TryGetInt32(tensor.Strides[1], out int stride1))
            {
                view = default;
                return false;
            }

            view = new GgmlTensorView2D(GetBufferStart(tensor), dim0, dim1, stride0, stride1, rawBytes);
            return true;
        }

        private static bool TryCreateRawView(Tensor tensor, out GgmlTensorView3D view)
        {
            if (tensor.DimensionCount != 3
                || tensor.ElementType != DType.Float32
                || !(tensor.Storage is GgmlStorage)
                || !TryGetRawSpanBytes(tensor, out long rawBytes)
                || !TryGetInt32(tensor.Sizes[0], out int dim0)
                || !TryGetInt32(tensor.Sizes[1], out int dim1)
                || !TryGetInt32(tensor.Sizes[2], out int dim2)
                || !TryGetInt32(tensor.Strides[0], out int stride0)
                || !TryGetInt32(tensor.Strides[1], out int stride1)
                || !TryGetInt32(tensor.Strides[2], out int stride2))
            {
                view = default;
                return false;
            }

            view = new GgmlTensorView3D(GetBufferStart(tensor), dim0, dim1, dim2, stride0, stride1, stride2, rawBytes);
            return true;
        }

        private static bool CanMapTensorToStandardGgmlView(Tensor tensor)
        {
            if (tensor.ElementType != DType.Float32
                || tensor.DimensionCount < 1
                || tensor.DimensionCount > 4
                || tensor.Strides[tensor.DimensionCount - 1] != 1)
            {
                return false;
            }

            long requiredStride = 1;
            for (int dimension = tensor.DimensionCount - 1; dimension >= 0; --dimension)
            {
                long size = tensor.Sizes[dimension];
                if (size <= 0)
                {
                    return false;
                }

                if (size == 1)
                {
                    continue;
                }

                long stride = tensor.Strides[dimension];
                if (stride < requiredStride)
                {
                    return false;
                }

                try
                {
                    requiredStride = checked(stride * size);
                }
                catch (OverflowException)
                {
                    return false;
                }
            }

            return true;
        }

        private static bool TryCreateCompactedBroadcastTensor(Tensor tensor, out Tensor compactTensor)
        {
            compactTensor = null;

            if (tensor.ElementType != DType.Float32
                || !(tensor.Storage is GgmlStorage)
                || tensor.DimensionCount < 1
                || tensor.DimensionCount > 4
                || tensor.Strides[tensor.DimensionCount - 1] != 1)
            {
                return false;
            }

            long[] compactSizes = (long[])tensor.Sizes.Clone();
            long[] compactStrides = (long[])tensor.Strides.Clone();
            bool changed = false;

            for (int i = 0; i < compactSizes.Length; ++i)
            {
                if (compactSizes[i] <= 0)
                {
                    return false;
                }

                if (compactSizes[i] > 1 && compactStrides[i] == 0)
                {
                    compactSizes[i] = 1;
                    changed = true;
                }
            }

            if (!changed)
            {
                return false;
            }

            compactTensor = new Tensor(compactSizes, compactStrides, tensor.Storage, tensor.StorageOffset);
            if (!CanMapTensorToStandardGgmlView(compactTensor))
            {
                compactTensor.Dispose();
                compactTensor = null;
                return false;
            }

            return true;
        }

        private static void ValidateMaskResultTensor(Tensor result, string opName)
        {
            ValidateGgmlTensor(result, nameof(result), opName);

            if (result.DimensionCount < 1)
            {
                throw new InvalidOperationException($"{opName} requires a tensor with at least one dimension.");
            }

            if (!result.IsContiguous())
            {
                throw new NotSupportedException($"{opName} currently requires a contiguous result tensor.");
            }
        }

        private static void ValidateMaskLengthsTensor(Tensor tensor, string argumentName, string opName)
        {
            ValidateGgmlTensor(tensor, argumentName, opName);

            if (!tensor.IsContiguous())
            {
                throw new NotSupportedException($"{opName} currently requires contiguous length tensors.");
            }
        }

        private static void ValidateGatherArguments(Tensor result, Tensor src, int dim, Tensor indices)
        {
            ValidateGgmlTensor(src, nameof(src), "gather");
            ValidateGgmlTensor(indices, nameof(indices), "gather");

            if (dim < 0 || dim >= src.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("gather expects src and indices to have the same number of dimensions.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "gather");
                if (result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException("gather expects result and src to have the same number of dimensions.");
                }

                if (!result.IsSameSizeAs(indices))
                {
                    throw new InvalidOperationException("gather expects result and indices to have the same shape.");
                }

                if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
                {
                    throw new InvalidOperationException("gather expects result and src to match in every dimension except dim.");
                }
            }
        }

        private static void ValidateScatterArguments(Tensor result, Tensor src, int dim, Tensor indices, string opName)
        {
            ValidateGgmlTensor(result, nameof(result), opName);
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlTensor(indices, nameof(indices), opName);

            if (dim < 0 || dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (result.DimensionCount != src.DimensionCount || indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException($"{opName} expects result, src, and indices to have the same number of dimensions.");
            }

            if (!src.IsSameSizeAs(indices))
            {
                throw new InvalidOperationException($"{opName} expects src and indices to have the same shape.");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException($"{opName} expects result and src to match in every dimension except dim.");
            }
        }

        private static void ValidateScatterFillArguments(Tensor result, int dim, Tensor indices)
        {
            ValidateGgmlTensor(result, nameof(result), "scatter_fill");
            ValidateGgmlTensor(indices, nameof(indices), "scatter_fill");

            if (dim < 0 || dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            if (indices.DimensionCount != result.DimensionCount)
            {
                throw new InvalidOperationException("scatter_fill expects result and indices to have the same number of dimensions.");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(indices.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("scatter_fill expects result and indices to match in every dimension except dim.");
            }
        }

        private static void ValidateElementwiseArguments(Tensor result, string opName, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException($"{opName} requires at least one tensor argument.", nameof(tensors));
            }

            Tensor reference = tensors[0];
            ValidateGgmlTensor(reference, "tensor0", opName);

            for (int i = 1; i < tensors.Length; ++i)
            {
                ValidateGgmlTensor(tensors[i], $"tensor{i}", opName);
                if (!HasSameShape(reference, tensors[i]))
                {
                    throw new InvalidOperationException($"{opName} expects all tensor arguments to have the same shape.");
                }
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, reference))
                {
                    throw new InvalidOperationException($"{opName} expects result to have the same shape as its tensor inputs.");
                }
            }
        }

        private static void GetFlatRowsCols(Tensor tensor, string opName, out int rows, out int cols)
        {
            long colsLong = tensor.Sizes[tensor.DimensionCount - 1];
            long storageSize = TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides);
            if (colsLong <= 0 || (storageSize % colsLong) != 0)
            {
                throw new InvalidOperationException($"{opName} received an invalid tensor layout.");
            }

            long rowsLong = storageSize / colsLong;
            if (!TryGetInt32(rowsLong, out rows) || !TryGetInt32(colsLong, out cols))
            {
                throw new NotSupportedException($"{opName} tensor dimensions exceed the GGML mask builder limits.");
            }
        }

        private static bool TryGetRawSpanBytes(Tensor tensor, out long rawBytes)
        {
            try
            {
                long maxOffset = 0;
                for (int dimension = 0; dimension < tensor.DimensionCount; ++dimension)
                {
                    long size = tensor.Sizes[dimension];
                    if (size <= 0)
                    {
                        rawBytes = 0;
                        return false;
                    }

                    maxOffset = checked(maxOffset + checked((size - 1) * tensor.Strides[dimension]));
                }

                rawBytes = checked((maxOffset + 1) * tensor.ElementType.Size());
                return true;
            }
            catch (OverflowException)
            {
                rawBytes = 0;
                return false;
            }
        }

        private static bool CanUseHostElementwiseWrite(Tensor tensor)
        {
            for (int dimension = 0; dimension < tensor.DimensionCount; ++dimension)
            {
                if (tensor.Sizes[dimension] > 1 && tensor.Strides[dimension] == 0)
                {
                    return false;
                }
            }

            return true;
        }

        private static bool TryGetInt32(long value, out int intValue)
        {
            try
            {
                intValue = checked((int)value);
                return true;
            }
            catch (OverflowException)
            {
                intValue = 0;
                return false;
            }
        }

        private static IntPtr GetBufferStart(Tensor tensor)
        {
            return ((GgmlStorage)tensor.Storage).PtrAtElement(tensor.StorageOffset);
        }

        private static void ValidateAddmmArguments(Tensor result, Tensor src, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type. src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result?.ElementType}'");
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }

            if (!(m1.Storage is GgmlStorage))
            {
                throw new ArgumentException("m1 must be a GGML tensor", nameof(m1));
            }

            if (!(m2.Storage is GgmlStorage))
            {
                throw new ArgumentException("m2 must be a GGML tensor", nameof(m2));
            }

            if (src.DimensionCount != 2 || m1.DimensionCount != 2 || m2.DimensionCount != 2)
            {
                throw new ArgumentException("addmm expects 2D tensors.");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
            {
                throw new InvalidOperationException("Size mismatch");
            }
        }

        private static void ValidateAddmmBatchArguments(Tensor result, Tensor src, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type. src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result?.ElementType}'");
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }

            if (!(m1.Storage is GgmlStorage))
            {
                throw new ArgumentException("m1 must be a GGML tensor", nameof(m1));
            }

            if (!(m2.Storage is GgmlStorage))
            {
                throw new ArgumentException("m2 must be a GGML tensor", nameof(m2));
            }

            if (src.DimensionCount != 3 || m1.DimensionCount != 3 || m2.DimensionCount != 3)
            {
                throw new ArgumentException("addmmbatch expects 3D tensors.");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[0] != m2.Sizes[0] || src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException("Size mismatch");
            }
        }

        private static void ValidateSoftmaxArguments(Tensor result, Tensor src)
        {
            if (src.ElementType != DType.Float32 || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"softmax expects Float32 tensors. src = '{src.ElementType}', result = '{result?.ElementType}'");
            }

            if (!(src.Storage is GgmlStorage))
            {
                throw new ArgumentException("src must be a GGML tensor", nameof(src));
            }

            if (result != null && !(result.Storage is GgmlStorage))
            {
                throw new ArgumentException("result must be a GGML tensor", nameof(result));
            }
        }

        private static void ValidateSoftmaxGradArguments(Tensor grad, Tensor adj, Tensor val)
        {
            if (adj.ElementType != DType.Float32 || val.ElementType != DType.Float32 || (grad != null && grad.ElementType != DType.Float32))
            {
                throw new InvalidOperationException($"softmaxgrad expects Float32 tensors. grad = '{grad?.ElementType}', adj = '{adj.ElementType}', val = '{val.ElementType}'");
            }

            if (!(adj.Storage is GgmlStorage))
            {
                throw new ArgumentException("adj must be a GGML tensor", nameof(adj));
            }

            if (!(val.Storage is GgmlStorage))
            {
                throw new ArgumentException("val must be a GGML tensor", nameof(val));
            }

            if (grad != null && !(grad.Storage is GgmlStorage))
            {
                throw new ArgumentException("grad must be a GGML tensor", nameof(grad));
            }

            if (adj.DimensionCount != val.DimensionCount)
            {
                throw new InvalidOperationException("adj and val must have the same number of dimensions.");
            }

            if (!HasSameShape(adj, val))
            {
                throw new InvalidOperationException("adj and val must have the same shape.");
            }

            if (grad != null && !HasSameShape(grad, adj))
            {
                throw new InvalidOperationException("grad and adj must have the same shape.");
            }
        }

        private static void ValidateAdamArguments(Tensor tw, Tensor tg, Tensor tv, Tensor tm)
        {
            if (tw.ElementType != DType.Float32 || tg.ElementType != DType.Float32 || tv.ElementType != DType.Float32 || tm.ElementType != DType.Float32)
            {
                throw new InvalidOperationException($"adam expects Float32 tensors. weight = '{tw.ElementType}', gradient = '{tg.ElementType}', v = '{tv.ElementType}', m = '{tm.ElementType}'");
            }

            if (!(tw.Storage is GgmlStorage))
            {
                throw new ArgumentException("weight must be a GGML tensor", nameof(tw));
            }

            if (!(tg.Storage is GgmlStorage))
            {
                throw new ArgumentException("gradient must be a GGML tensor", nameof(tg));
            }

            if (!(tv.Storage is GgmlStorage))
            {
                throw new ArgumentException("v must be a GGML tensor", nameof(tv));
            }

            if (!(tm.Storage is GgmlStorage))
            {
                throw new ArgumentException("m must be a GGML tensor", nameof(tm));
            }

            if (!HasSameShape(tw, tg)
                || !HasSameShape(tw, tv)
                || !HasSameShape(tw, tm))
            {
                throw new InvalidOperationException("weight, gradient, v, and m must have the same shape.");
            }
        }

        private static void ValidateCopyArguments(Tensor result, Tensor src)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            ValidateGgmlTensor(result, nameof(result), "copy");
            ValidateGgmlTensor(src, nameof(src), "copy");

            if (result.ElementCount() != src.ElementCount())
            {
                throw new InvalidOperationException("copy expects source and result to have the same number of elements.");
            }
        }

        private static unsafe Tensor ExecuteReduction(Tensor result, Tensor src, int dimension, bool divideByCount, string opName)
        {
            ValidateReductionArguments(result, src, dimension, opName);

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeReduction(writeTarget, src, dimension, divideByCount ? GgmlReductionOp.Mean : GgmlReductionOp.Sum))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dimension);

            do
            {
                float sum = 0.0f;
                for (long i = 0; i < srcIter.size; ++i)
                {
                    sum += *(srcIter.data + i * srcIter.stride);
                }

                *resultIter.data = divideByCount ? (sum / srcIter.size) : sum;
            } while (resultIter.NextBlock() && srcIter.NextBlock());

            return writeTarget;
        }

        private static unsafe Tensor ExecuteIndexReduction(Tensor result, Tensor src, int dimension, bool selectMinimum, string opName)
        {
            ValidateReductionArguments(result, src, dimension, opName);

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);
            if (TryExecuteNativeIndexReduction(writeTarget, src, dimension, selectMinimum ? GgmlIndexReductionOp.Argmin : GgmlIndexReductionOp.Argmax))
            {
                return writeTarget;
            }

            TensorDimIterState resultIter = new TensorDimIterState((float*)GetBufferStart(writeTarget), writeTarget.DimensionCount, writeTarget.Sizes, writeTarget.Strides, dimension);
            TensorDimIterState srcIter = new TensorDimIterState((float*)GetBufferStart(src), src.DimensionCount, src.Sizes, src.Strides, dimension);

            do
            {
                long bestIndex = 0;
                float bestValue = *srcIter.data;
                for (long i = 1; i < srcIter.size; ++i)
                {
                    float currentValue = *(srcIter.data + i * srcIter.stride);
                    if (selectMinimum ? currentValue < bestValue : currentValue > bestValue)
                    {
                        bestValue = currentValue;
                        bestIndex = i;
                    }
                }

                *resultIter.data = bestIndex;
            } while (resultIter.NextBlock() && srcIter.NextBlock());

            return writeTarget;
        }

        private static bool TryExecuteNativeReduction(Tensor result, Tensor src, int dimension, GgmlReductionOp op)
        {
            if (dimension != src.DimensionCount - 1)
            {
                return false;
            }

            if (!TryCreateStandardView(result, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                return false;
            }

            GgmlNative.ReduceLastDim(op, resultView, srcView);
            return true;
        }

        private static bool TryExecuteNativeIndexReduction(Tensor result, Tensor src, int dimension, GgmlIndexReductionOp op)
        {
            if (dimension != src.DimensionCount - 1)
            {
                return false;
            }

            if (!TryCreateStandardView(result, out GgmlTensorView4D resultView)
                || !TryCreateStandardView(src, out GgmlTensorView4D srcView))
            {
                return false;
            }

            GgmlNative.IndexReduction(op, resultView, srcView);
            return true;
        }

        private static void ValidateReductionArguments(Tensor result, Tensor src, int dimension, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (dimension < 0 || dimension >= src.DimensionCount)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            if (src.Sizes[dimension] <= 0)
            {
                throw new InvalidOperationException($"{opName} expects a non-empty reduction dimension.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);

                if (result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same number of dimensions.");
                }

                for (int i = 0; i < src.DimensionCount; ++i)
                {
                    long expected = i == dimension ? 1 : src.Sizes[i];
                    if (result.Sizes[i] != expected)
                    {
                        throw new InvalidOperationException($"{opName} expects result to match src except for a singleton reduction dimension.");
                    }
                }
            }
        }

        private static void ValidateUnaryArguments(Tensor result, Tensor src, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateBinaryTensorArguments(Tensor result, Tensor lhs, Tensor rhs, string opName)
        {
            ValidateGgmlTensor(lhs, nameof(lhs), opName);
            ValidateGgmlTensor(rhs, nameof(rhs), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, lhs))
                {
                    throw new InvalidOperationException($"{opName} expects result and lhs to have the same shape.");
                }
            }
        }

        private static void ValidateBinaryScalarArguments(Tensor result, Tensor src, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateActivationGradArguments(Tensor result, Tensor accumulation, Tensor src, Tensor grad, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlTensor(grad, nameof(grad), opName);

            if (!HasSameShape(src, grad))
            {
                throw new InvalidOperationException($"{opName} expects src and grad to have the same shape.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }

            if (accumulation != null)
            {
                ValidateGgmlTensor(accumulation, nameof(accumulation), opName);
                if (!HasSameShape(accumulation, src))
                {
                    throw new InvalidOperationException($"{opName} expects accumulation and src to have the same shape.");
                }
            }
        }

        private static void ValidateNormArguments(Tensor result, Tensor src, Tensor gamma, Tensor beta, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);
            ValidateGgmlTensor(gamma, nameof(gamma), opName);

            if (src.DimensionCount < 2 || src.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if (gamma.ElementCount() != src.Sizes[src.DimensionCount - 1])
            {
                throw new InvalidOperationException($"{opName} expects gamma element count to match the last dimension of src.");
            }

            if (beta != null)
            {
                ValidateGgmlTensor(beta, nameof(beta), opName);
                if (beta.ElementCount() != src.Sizes[src.DimensionCount - 1])
                {
                    throw new InvalidOperationException($"{opName} expects beta element count to match the last dimension of src.");
                }
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateNormGradArguments(Tensor result, Tensor gradGamma, Tensor gradBeta, Tensor adj, Tensor y, Tensor x, Tensor gamma, Tensor beta, string opName)
        {
            ValidateGgmlTensor(adj, nameof(adj), opName);
            ValidateGgmlTensor(y, nameof(y), opName);
            ValidateGgmlTensor(x, nameof(x), opName);
            ValidateGgmlTensor(gamma, nameof(gamma), opName);
            ValidateGgmlTensor(gradGamma, nameof(gradGamma), opName);

            if (x.DimensionCount < 2 || x.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if (!HasSameShape(adj, x) || !HasSameShape(y, x))
            {
                throw new InvalidOperationException($"{opName} expects adj, y, and x to have the same shape.");
            }

            if (gamma.ElementCount() != x.Sizes[x.DimensionCount - 1])
            {
                throw new InvalidOperationException($"{opName} expects gamma element count to match the last dimension of x.");
            }

            if (!HasSameShape(gradGamma, gamma))
            {
                throw new InvalidOperationException($"{opName} expects gradGamma to have the same shape as gamma.");
            }

            if (beta != null)
            {
                ValidateGgmlTensor(beta, nameof(beta), opName);
                if (beta.ElementCount() != x.Sizes[x.DimensionCount - 1])
                {
                    throw new InvalidOperationException($"{opName} expects beta element count to match the last dimension of x.");
                }

                if (gradBeta == null)
                {
                    throw new ArgumentNullException(nameof(gradBeta), $"{opName} requires gradBeta when beta is provided.");
                }

                ValidateGgmlTensor(gradBeta, nameof(gradBeta), opName);
                if (!HasSameShape(gradBeta, beta))
                {
                    throw new InvalidOperationException($"{opName} expects gradBeta to have the same shape as beta.");
                }
            }
            else if (gradBeta != null)
            {
                throw new InvalidOperationException($"{opName} does not accept gradBeta when beta is null.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, x))
                {
                    throw new InvalidOperationException($"{opName} expects result and x to have the same shape.");
                }
            }
        }

        private static void ValidateIndexSelectArguments(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            ValidateGgmlTensor(src, nameof(src), "indexselect");
            ValidateGgmlTensor(indice, nameof(indice), "indexselect");

            if (isAdd && result == null)
            {
                throw new ArgumentNullException(nameof(result), "indexselect with isAdd=true requires an existing result tensor.");
            }

            if (src.DimensionCount != 2)
            {
                throw new NotSupportedException("GGML indexselect currently supports 2D source tensors only.");
            }

            if (!IsSupportedIndexTensor(indice))
            {
                throw new NotSupportedException("GGML indexselect currently supports contiguous 1D or Nx1 index tensors only.");
            }

            if (!indice.IsContiguous())
            {
                throw new NotSupportedException("GGML indexselect requires contiguous indices.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), "indexselect");
                if (result.DimensionCount != 2 || result.Sizes[0] != indice.Sizes[0] || result.Sizes[1] != src.Sizes[1])
                {
                    throw new InvalidOperationException("indexselect expects result shape [indices, src_cols].");
                }
            }
        }

        private static void ValidateIndexSelectGradArguments(Tensor grad, Tensor adj, Tensor indice)
        {
            ValidateGgmlTensor(grad, nameof(grad), "indexselectgrad");
            ValidateGgmlTensor(adj, nameof(adj), "indexselectgrad");
            ValidateGgmlTensor(indice, nameof(indice), "indexselectgrad");

            if (grad.DimensionCount != 2 || adj.DimensionCount != 2)
            {
                throw new NotSupportedException("GGML indexselectgrad currently supports 2D gradient and adjoint tensors only.");
            }

            if (!IsSupportedIndexTensor(indice))
            {
                throw new NotSupportedException("GGML indexselectgrad currently supports contiguous 1D or Nx1 index tensors only.");
            }

            if (!indice.IsContiguous())
            {
                throw new NotSupportedException("GGML indexselectgrad requires contiguous indices.");
            }

            if (adj.Sizes[0] != indice.Sizes[0] || grad.Sizes[1] != adj.Sizes[1])
            {
                throw new InvalidOperationException("indexselectgrad expects adj shape [indices, grad_cols].");
            }
        }

        private static void ValidateRoPEArguments(Tensor result, Tensor src, int seqLen, string opName)
        {
            ValidateGgmlTensor(src, nameof(src), opName);

            if (seqLen <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen), $"{opName} requires seqLen > 0.");
            }

            if (src.DimensionCount < 2 || src.DimensionCount > 4)
            {
                throw new NotSupportedException($"{opName} currently supports 2D to 4D tensors only.");
            }

            if ((src.Sizes[src.DimensionCount - 1] & 1) != 0)
            {
                throw new NotSupportedException($"{opName} requires the last tensor dimension to be even.");
            }

            if (result != null)
            {
                ValidateGgmlTensor(result, nameof(result), opName);
                if (!HasSameShape(result, src))
                {
                    throw new InvalidOperationException($"{opName} expects result and src to have the same shape.");
                }
            }
        }

        private static void ValidateGgmlTensor(Tensor tensor, string argumentName, string opName)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(argumentName);
            }

            if (tensor.ElementType != DType.Float32)
            {
                throw new InvalidOperationException($"{opName} expects Float32 tensors only.");
            }

            if (!(tensor.Storage is GgmlStorage))
            {
                throw new ArgumentException($"{argumentName} must be a GGML tensor", argumentName);
            }
        }

        private static bool HasSameShape(Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount != rhs.DimensionCount)
            {
                return false;
            }

            for (int i = 0; i < lhs.DimensionCount; ++i)
            {
                if (lhs.Sizes[i] != rhs.Sizes[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static bool AreEquivalentViews(Tensor lhs, Tensor rhs)
        {
            if (!ReferenceEquals(lhs.Storage, rhs.Storage) || lhs.StorageOffset != rhs.StorageOffset)
            {
                return false;
            }

            if (lhs.DimensionCount != rhs.DimensionCount)
            {
                return false;
            }

            for (int i = 0; i < lhs.DimensionCount; ++i)
            {
                if (lhs.Sizes[i] != rhs.Sizes[i] || lhs.Strides[i] != rhs.Strides[i])
                {
                    return false;
                }
            }

            return true;
        }

        private static bool HasExpandedWriteDimension(Tensor tensor)
        {
            for (int i = 0; i < tensor.DimensionCount; ++i)
            {
                if (tensor.Sizes[i] > 1 && tensor.Strides[i] == 0)
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsSupportedIndexTensor(Tensor indice)
        {
            if (indice.DimensionCount == 1)
            {
                return true;
            }

            if (indice.DimensionCount == 2 && indice.Sizes[1] == 1)
            {
                return true;
            }

            return false;
        }
    }
}
