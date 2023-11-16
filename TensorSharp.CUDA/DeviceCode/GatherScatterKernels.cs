using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Text;
using TensorSharp.Core;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class GatherScatterKernels : CudaCode
    {
        public static readonly string Code = @"
extern ""C"" {\
__global__ void gather_kernel(
    TensorInfo<unsigned __int64> tensor,
    TensorInfo<unsigned __int64> src,
    TensorInfo<unsigned __int64> index,
    const int dim,
    const unsigned __int64 totalElements) {
  for (unsigned __int64 linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    unsigned __int64 tensorOffset = 0;
    unsigned __int64 srcOffset = 0;
    unsigned __int64 indexOffset = 0;

    unsigned __int64 linearId2 = linearId;
    for (int d = index.dims - 1; d >= 0; d--) {
      unsigned __int64 curDimIndex = linearId2 % index.sizes[d];
      indexOffset += curDimIndex * index.strides[d];
      tensorOffset += curDimIndex * tensor.strides[d];
      if (d != dim) {
        srcOffset += curDimIndex * src.strides[d];
      }
      linearId2 /= index.sizes[d];
    }

    unsigned __int64 indexValue = (unsigned __int64)index.data[indexOffset];
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}
}

extern ""C"" {\
__global__ void scatter_kernel(
    TensorInfo<unsigned __int64> tensor,
    TensorInfo<unsigned __int64> src,
    TensorInfo<unsigned __int64> index,
    const int dim,
    const unsigned __int64 totalElements) {
  for (unsigned __int64 linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    unsigned __int64 tensorOffset = 0;
    unsigned __int64 srcOffset = 0;
    unsigned __int64 indexOffset = 0;

    unsigned __int64 linearId2 = linearId;
    for (int d = index.dims - 1; d >= 0; d--) {
      unsigned __int64 curDimIndex = linearId2 % index.sizes[d];
      indexOffset += curDimIndex * index.strides[d];
      srcOffset += curDimIndex * src.strides[d];
      if (d != dim) {
        tensorOffset += curDimIndex * tensor.strides[d];
      }
      linearId2 /= index.sizes[d];
    }

    unsigned __int64 indexValue = (unsigned __int64)index.data[indexOffset];
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}
}


extern ""C"" {\
__global__ void scatterAdd_kernel(
    TensorInfo<unsigned __int64> tensor,
    TensorInfo<unsigned __int64> src,
    TensorInfo<unsigned __int64> index,
    const int dim,
    const unsigned __int64 totalElements) {
  for (unsigned __int64 linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    unsigned __int64 tensorOffset = 0;
    unsigned __int64 srcOffset = 0;
    unsigned __int64 indexOffset = 0;

    unsigned __int64 linearId2 = linearId;
    for (int d = index.dims - 1; d >= 0; d--) {
      unsigned __int64 curDimIndex = linearId2 % index.sizes[d];
      indexOffset += curDimIndex * index.strides[d];
      srcOffset += curDimIndex * src.strides[d];
      if (d != dim) {
        tensorOffset += curDimIndex * tensor.strides[d];
      }
      linearId2 /= index.sizes[d];
    }

    unsigned __int64 indexValue = (unsigned __int64)index.data[indexOffset];
    tensorOffset += indexValue * tensor.strides[dim];


    atomicAdd(tensor.data + tensorOffset, src.data[srcOffset]);

    
  }
}
}

extern ""C"" {\
__global__ void scatterFill_kernel(
    TensorInfo<unsigned __int64> tensor,
    TensorInfo<unsigned __int64> index,
    float value,
    const int dim,
    const unsigned __int64 totalElements) {
  for (unsigned __int64 linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    unsigned __int64 tensorOffset = 0;
    unsigned __int64 indexOffset = 0;

    unsigned __int64 linearId2 = linearId;
    for (int d = index.dims - 1; d >= 0; d--) {
      unsigned __int64 curDimIndex = linearId2 % index.sizes[d];
      indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        tensorOffset += curDimIndex * tensor.strides[d];
      }
      linearId2 /= index.sizes[d];
    }

    unsigned __int64 indexValue = (unsigned __int64)index.data[indexOffset];
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = value;
  }
}
} 
";

        public GatherScatterKernels() : base(GetCode(), "General", "ReduceApplyUtils")
        {
        }


        private static string GetCode()
        {
            StringBuilder sb = new StringBuilder(Code);
            return sb.ToString();
        }

        public Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            try
            {
                TSCudaContext context = CudaHelpers.TSContextForTensor(src);
                CudaContext cudaContext = context.CudaContextForTensor(src);

                if (result != null && result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException($"result and src must have same number of dimensions. result dim count = '{result.DimensionCount}', src dim count = '{src.DimensionCount}'");
                }

                if (result != null && dim < 0 && dim >= result.DimensionCount)
                {
                    throw new ArgumentOutOfRangeException("dim");
                }

                if (indices.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException($"src and indices must have same number of dimensions. indices dim count = '{indices.DimensionCount}', src dim count = '{src.DimensionCount}'");
                }

                if (result != null && !result.IsSameSizeAs(indices))
                {
                    throw new InvalidOperationException($"result and indices must be the same size. result = '{result.ToString()}', indices = '{indices.ToString()}'");
                }

                if (result != null && !TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
                {
                    throw new InvalidOperationException($"result and src must be the same size except in dimension dim. src = '{src.ToString()}', result = '{result.ToString()}', dim = '{dim}'");
                }

                Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);

                long nElement = indices.ElementCount();
                dim3 block = ApplyUtils.GetApplyBlock();
                dim3 grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

                Invoke(context, cudaContext, "gather_kernel", grid, block, 0, CUstream.NullStream, false, writeTarget, src, indices, dim, nElement);

                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Error = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack = '{err.StackTrace}'");
                throw;
            }
        }

        public Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            try
            {
                TSCudaContext context = CudaHelpers.TSContextForTensor(src);
                CudaContext cudaContext = context.CudaContextForTensor(src);

                if (result == null)
                {
                    throw new ArgumentNullException("result");
                }

                if (result.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException($"result and src must have same number of dimensions. result dim count = '{result.DimensionCount}', source dim count = '{src.DimensionCount}'");
                }

                if (dim < 0 && dim >= result.DimensionCount)
                {
                    throw new ArgumentOutOfRangeException("dim");
                }

                if (indices.DimensionCount != src.DimensionCount)
                {
                    throw new InvalidOperationException("src and indices must have same number of dimensions");
                }

                if (!src.IsSameSizeAs(indices))
                {
                    throw new InvalidOperationException("src and indices must be the same size");
                }

                if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
                {
                    throw new InvalidOperationException("result and src must be the same size except in dimension dim");
                }

                Tensor writeTarget = result;

                long nElement = indices.ElementCount();
                dim3 block = ApplyUtils.GetApplyBlock();
                dim3 grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

                Invoke(context, cudaContext, "scatter_kernel", grid, block, 0, CUstream.NullStream, false, writeTarget, src, indices, dim, nElement);

                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Error = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call stack = '{err.StackTrace}'");
                throw;
            }
        }


        public Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            CudaContext cudaContext = context.CudaContextForTensor(src);

            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (result.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("result and src must have same number of dimensions");
            }

            if (dim < 0 && dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dim");
            }

            if (indices.DimensionCount != src.DimensionCount)
            {
                throw new InvalidOperationException("src and indices must have same number of dimensions");
            }

            if (!src.IsSameSizeAs(indices))
            {
                throw new InvalidOperationException("src and indices must be the same size");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("result and src must be the same size except in dimension dim");
            }

            Tensor writeTarget = result;

            long nElement = indices.ElementCount();
            dim3 block = ApplyUtils.GetApplyBlock();
            dim3 grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            Invoke(context, cudaContext, "scatterAdd_kernel", grid, block, 0, CUstream.NullStream, false, writeTarget, src, indices, dim, nElement);

            return writeTarget;
        }

        public Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(indices);
            CudaContext cudaContext = context.CudaContextForTensor(indices);

            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (dim < 0 && dim >= result.DimensionCount)
            {
                throw new ArgumentOutOfRangeException("dim");
            }

            if (indices.DimensionCount != result.DimensionCount)
            {
                throw new InvalidOperationException("result and indices must have same number of dimensions");
            }

            if (!TensorResultBuilder.ArrayEqualExcept(indices.Sizes, result.Sizes, dim))
            {
                throw new InvalidOperationException("result and indices must be the same size except in dimension dim");
            }

            Tensor writeTarget = result;

            long nElement = indices.ElementCount();
            dim3 block = ApplyUtils.GetApplyBlock();
            dim3 grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            Invoke(context, cudaContext, "scatterFill_kernel", grid, block, 0, CUstream.NullStream, false,
               writeTarget, indices, value, dim, nElement);

            return writeTarget;
        }


        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, bool index32, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, index32, args);

            byte[] ptx = GetPtx(context.Compiler);
            CudaKernel kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
