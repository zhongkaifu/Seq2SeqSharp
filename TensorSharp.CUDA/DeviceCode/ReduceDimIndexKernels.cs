using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using TensorSharp.Core;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ReduceDimIndexKernels : CudaCode
    {
        private static readonly string Code = @"

REDUCE_INDEX_KERNELS(argmin, if (a.first < b.first) return a; else return b;)
REDUCE_INDEX_KERNELS(argmax, if (a.first > b.first) return a; else return b;)

";

        public ReduceDimIndexKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "ReduceIndex", "Math")
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }

        private void ReduceIndexOuterDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            int ndim = src.DimensionCount;
            long num_orows = 1;
            for (int dim = 0; dim < dimension; dim++)
            {
                num_orows *= src.Sizes[dim];
            }
            long row_size = src.Sizes[dimension];
            long num_irows = 1;
            for (int dim = dimension + 1; dim < ndim; dim++)
            {
                num_irows *= src.Sizes[dim];
            }

            dim3 threads = new dim3((uint)Math.Min(512, num_irows));
            int maxGridDim = 1024;
            dim3 grid = new dim3((uint)Math.Min(maxGridDim, num_orows), (uint)Math.Min(maxGridDim, ApplyUtils.CeilDiv(num_irows, threads.x)));

            CUdeviceptr resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            CUdeviceptr resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);

            string kernelName = "outer_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_orows, num_irows, row_size, init.Item1, init.Item2);
        }

        private void ReduceIndexInnermostDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, Tuple<float, float> init, string baseKernelName)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            int ndim = src.DimensionCount;
            long num_rows = 1;
            for (int dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Sizes[dim];
            }
            long row_size = src.Sizes[ndim - 1];

            dim3 threads = new dim3(16, 32);
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            CUdeviceptr resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            CUdeviceptr resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);

            string kernelName = "inner_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_rows, row_size, init.Item1, init.Item2);
        }

        private Tensor RunReduceIndexOp(Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            long[] requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(resultIndices, src.Allocator, DType.Float32, true, requiredOutputSize);

            using (Tensor resultValueBuffer = new Tensor(src.Allocator, src.ElementType, requiredOutputSize))
            {
                if (dimension == src.DimensionCount - 1)
                {
                    ReduceIndexInnermostDim(context, resultValueBuffer, writeTarget, src, init, baseKernelName);
                }
                else
                {
                    ReduceIndexOuterDim(context, resultValueBuffer, writeTarget, src, dimension, init, baseKernelName);
                }

            }

            return writeTarget;
        }

        public Tensor ArgMin(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MaxValue, 0.0f), "argmin");
        }

        public Tensor ArgMax(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MinValue, 0.0f), "argmax");
        }

        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            byte[] ptx = GetPtx(context.Compiler);
            CudaKernel kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
