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
    class AdvFuncKernels : CudaCode
    {
        private static readonly string Code = @"
extern ""C""
{

__global__ void gLNormalization(float* out,
                                const float* in,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * ((sp[id] - mean) / sigma);
          if(beta)
            t += beta[id];
          so[id] = t;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void gLayerNormalizationGrad(float* gradX,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += xRow[id];
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = xRow[id] - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = gamma[id] * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000 ? sign * 1000 : valX;

          gradXRow[id] += valX;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
    __syncthreads();
  }
}








__global__ void gAddLNormalization(float* out,
                                const float* in1,
                                const float* in2,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp1 = in1 + j * cols;
      const float* sp2 = in2 + j * cols;

      float* _sum = _share;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (sp1[id] + sp2[id]);
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = (sp1[id] + sp2[id]) - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * (((sp1[id] + sp2[id]) - mean) / sigma);
          if(beta)
            t += beta[id];
          so[id] = t;
        }
      }
    }
    __syncthreads();
  }
}






__global__ void gAddLayerNormalizationGrad(float* gradX1,
                                        float* gradX2,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x1,
                                        float* x2,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* x1Row = x1 + j * cols;
      const float* x2Row = x2 + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradX1Row = gradX1 + j * cols;
      float* gradX2Row = gradX2 + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += (x1Row[id] + x2Row[id]);
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = (x1Row[id] + x2Row[id]) - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = gamma[id] * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000 ? sign * 1000 : valX;

          gradX1Row[id] += valX;
          gradX2Row[id] += valX;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
    __syncthreads();
  }
}



__global__ void Adam(float* __restrict__ w, float* __restrict__ g, float* __restrict__ v, float* __restrict__ m, unsigned rows, unsigned cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    {
      float* sw = w + j * cols;
      float* sg = g + j * cols;
      float* sv = v + j * cols;
      float* sm = m + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols && sg[i] != 0.0) 
        {
           float g = sg[i] / batchSize;
           
           if (g > clipval)
           {
               g = clipval;
           }
           if (g < -clipval)
           {
               g = -clipval;
           }

           sm[i] = sm[i] * decay_rate_m + (1.0 - decay_rate_m) * g;
           sv[i] = sv[i] * decay_rate_v + (1.0 - decay_rate_v) * g * g;

           float m_cap = sm[i] / (1.0 - powf(decay_rate_m, iter));
           float v_cap = sv[i] / (1.0 - powf(decay_rate_v, iter));

           sw[i] -= step_size * m_cap / (sqrtf(v_cap) + eps);

           sg[i] = 0;
        }
      }
    }
  }
}




__global__ void RMSProp(float* __restrict__ w, float* __restrict__ g, float* __restrict__ c, unsigned rows, unsigned cols, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    {
      float* sw = w + j * cols;
      float* sg = g + j * cols;
      float* sc = c + j * cols;
      
      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols && sg[i] != 0.0) 
        {
           float g = sg[i] / batchSize;
           
           if (g > clipval)
           {
               g = clipval;
           }
           if (g < -clipval)
           {
               g = -clipval;
           }

           sc[i] = sc[i] * decay_rate + (1.0 - decay_rate) * g * g;

           g = g * rsqrtf(sc[i] + eps);

           sw[i] -= g * step_size + sw[i] * regc;

           sg[i] = 0;
        }
      }
    }
  }
}


  __global__ void gSoftmaxGrad(float* grad, float* adj, float* val, int rows, int cols, int addGrad)
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

          if (addGrad == 0)
          {
             gradRow[id] = v;
          }
          else
          {
             gradRow[id] += v;
          }
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
          float ex = expf(sp[i] - max);
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

        public AdvFuncKernels()
            : base(GetFullCode())
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }


        public Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(inGrad);
            var writeTarget = TensorResultBuilder.GetWriteTarget(outGrad, inGrad, false, inGrad.Sizes);
            LayerNormGrad(context, writeTarget, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNormGrad(TSCudaContext context, Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();

            var rows = inGrad.Sizes[0];
            var cols = inGrad.Sizes[1];

            var ndim = inGrad.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= inGrad.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var outGradPtr = CudaHelpers.GetBufferStart(outGrad);
            var alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            var betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            var inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            var yPtr = CudaHelpers.GetBufferStart(y);
            var xPtr = CudaHelpers.GetBufferStart(x);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gLayerNormalizationGrad", grid, threads, (uint)(threads.x * sizeof(float)) * 4, CUstream.NullStream, outGradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, xPtr,  alphaPtr, betaPtr, rows, cols, eps);

        }

        public void AddLayerNormGrad(Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(inGrad);
            var writeTarget1 = TensorResultBuilder.GetWriteTarget(out1Grad, inGrad, false, inGrad.Sizes);
            var writeTarget2 = TensorResultBuilder.GetWriteTarget(out2Grad, inGrad, false, inGrad.Sizes);
            AddLayerNormGrad(context, writeTarget1, writeTarget2, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps);
        }

        private void AddLayerNormGrad(TSCudaContext context, Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();

            var rows = inGrad.Sizes[0];
            var cols = inGrad.Sizes[1];

            var ndim = inGrad.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= inGrad.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var out1GradPtr = CudaHelpers.GetBufferStart(out1Grad);
            var out2GradPtr = CudaHelpers.GetBufferStart(out2Grad);
            var alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            var betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            var inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            var yPtr = CudaHelpers.GetBufferStart(y);
            var x1Ptr = CudaHelpers.GetBufferStart(x1);
            var x2Ptr = CudaHelpers.GetBufferStart(x2);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gAddLayerNormalizationGrad", grid, threads, (uint)(threads.x * sizeof(float)) * 4, CUstream.NullStream, out1GradPtr, out2GradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, x1Ptr, x2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }


        public Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            LayerNorm(context, writeTarget, src, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNorm(TSCudaContext context, Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

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
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gLNormalization", grid, threads, (uint)(threads.x * sizeof(float)), CUstream.NullStream, resultPtr, srcPtr, alphaPtr, betaPtr, rows, cols, eps);

        }

        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var context = CudaHelpers.TSContextForTensor(src1);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src1, false, src1.Sizes);
            AddLayerNorm(context, writeTarget, src1, src2, alpha, beta, eps);

            return writeTarget;
        }

        private void AddLayerNorm(TSCudaContext context, Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            var cudaContext = context.CudaContextForTensor(src1);

            cudaContext.SetCurrent();

            var rows = src1.Sizes[0];
            var cols = src1.Sizes[1];

            var ndim = src1.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src1.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var src1Ptr = CudaHelpers.GetBufferStart(src1);
            var src2Ptr = CudaHelpers.GetBufferStart(src2);
            var alphaPtr = CudaHelpers.GetBufferStart(alpha);
            var betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gAddLNormalization", grid, threads, (uint)(threads.x * sizeof(float)), CUstream.NullStream, resultPtr, src1Ptr, src2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }


        private void Softmax(TSCudaContext context, Tensor result, Tensor src)
        {
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

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

        private void Adam(TSCudaContext context, Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            var cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            var rows = weight.Sizes[0];
            var cols = weight.Sizes[1];

            var ndim = weight.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= weight.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var weightPtr = CudaHelpers.GetBufferStart(weight);
            var gradientPtr = CudaHelpers.GetBufferStart(gradient);
            var vPtr = CudaHelpers.GetBufferStart(v);
            var mPtr = CudaHelpers.GetBufferStart(m);

            Invoke(context, cudaContext, "Adam", grid, threads, 0, CUstream.NullStream, weightPtr, gradientPtr, vPtr, mPtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
        }

        public Tensor Adam(Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            var context = CudaHelpers.TSContextForTensor(weight);
            Adam(context, weight, gradient, v, m, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);

            return weight;
        }

        private void RMSProp(TSCudaContext context, Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            var cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            var rows = weight.Sizes[0];
            var cols = weight.Sizes[1];

            var ndim = weight.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= weight.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_rows));
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var weightPtr = CudaHelpers.GetBufferStart(weight);
            var gradientPtr = CudaHelpers.GetBufferStart(gradient);
            var cachePtr = CudaHelpers.GetBufferStart(cache);

            Invoke(context, cudaContext, "RMSProp", grid, threads, 0, CUstream.NullStream, weightPtr, gradientPtr, cachePtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate, eps);
        }

        public Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            var context = CudaHelpers.TSContextForTensor(weight);
            RMSProp(context, weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);

            return weight;
        }

        private void SoftmaxGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            var cudaContext = context.CudaContextForTensor(grad);

            cudaContext.SetCurrent();

            var rows = grad.Sizes[0];
            var cols = grad.Sizes[1];
            var iAddGrad = addGrad ? 1 : 0;

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

            Invoke(context, cudaContext, "gSoftmaxGrad", grid, threads, (uint)(threads.x * sizeof(float)), CUstream.NullStream, gradPtr, adjPtr, valPtr, rows, cols, iAddGrad);
        }


        public Tensor Softmax(Tensor result, Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            Softmax(context, writeTarget, src);

            return writeTarget;
        }


        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            var context = CudaHelpers.TSContextForTensor(grad);
            SoftmaxGrad(context, grad, adj, val, addGrad);

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
