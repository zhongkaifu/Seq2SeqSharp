using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    class SoftmaxKernels : CudaCode
    {
        private static readonly string Code = @"
extern ""C""
{
  __global__ void gSoftmaxGrad(float* grad, float* adj, float* val, int rows, int cols)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float v = valRow[id] * adjRow[id];
          gradRow[id] += v;
          _sum[threadIdx.x] += v;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
        //  float val = valRow[id] * (adjRow[id] - sum);
        //  if(val)
        //    gradRow[id] += val;

         gradRow[id] -= sum * valRow[id];
        }
      }
    }
    __syncthreads();
  }
}

  __global__ void gSoftmax(float* out, float* in, unsigned rows, unsigned cols)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];     
      float* _max = _share;
      _max[threadIdx.x] = -999999999.0; 
      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(sp[i] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[i];
        }
      }
      __syncthreads();      
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
            _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float max = _max[0];
      __syncthreads();
    
      float* _sum = _share;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {         
          float ex = __expf(sp[i] - max);
          so[i] = ex;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();     
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
    
      float sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          so[i] = so[i] / sum;
        }
      }
    }
    __syncthreads();
  }
}
}

";

        public SoftmaxKernels()
            : base(GetFullCode())
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }

     
        private void Softmax(TSCudaContext context, Tensor result, Tensor src)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var rows = src.Sizes[0];
            var cols = src.Sizes[1];

            var ndim = src.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            Invoke(context, cudaContext, "gSoftmax", grid, threads, (uint)(threads.x * sizeof(float)), CUstream.NullStream, resultPtr, srcPtr, rows, cols);
        }



        private void SoftmaxGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor val)
        {
            var cudaContext = context.CudaContextForTensor(grad);

            var rows = grad.Sizes[0];
            var cols = grad.Sizes[1];

            var ndim = grad.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= grad.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var gradPtr = CudaHelpers.GetBufferStart(grad);
            var adjPtr = CudaHelpers.GetBufferStart(adj);
            var valPtr = CudaHelpers.GetBufferStart(val);

            Invoke(context, cudaContext, "gSoftmaxGrad", grid, threads, (uint)(threads.x * sizeof(float)), CUstream.NullStream, gradPtr, adjPtr, valPtr, rows, cols);
        }


        public Tensor Softmax(Tensor result, Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            Softmax(context, writeTarget, src);

            return writeTarget;
        }


        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val)
        {
            var context = CudaHelpers.TSContextForTensor(grad);
            SoftmaxGrad(context, grad, adj, val);

            return grad;
        }

    
        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
