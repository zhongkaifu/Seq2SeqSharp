// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Runtime.InteropServices.Marshalling;
using TensorSharp.Core;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    internal class AdvFuncKernels : CudaCode
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
          float t = alpha[id] * (sp[id] - mean) / sigma;
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
          valX = fabs(valX) > 1000.0f ? sign * 1000.0f : valX;

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


__global__ void RMSNorm(float* out,
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

      float* _sqSum = _share;
      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id];
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      int len = blockDim.x;
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
          float t = alpha[id] * sp[id] / sigma;
          if(beta)
            t += beta[id];
          so[id] = t;
        }
      }
    }
    __syncthreads();
  }
}


__global__ void RMSNormGrad(float* gradX,
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
      float* sum_sqr = shared + 2 * blockDim.x;

      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
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
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = xRow[id];
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
          valX = fabs(valX) > 1000.0f ? sign * 1000.0f : valX;

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
          valX = fabs(valX) > 1000.0f ? sign * 1000.0f : valX;

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

      float bias_correction1 = 1.0 / (1.0 - powf(decay_rate_m, iter));
      float bias_correction2 = 1.0 / (1.0 - powf(decay_rate_v, iter));
      float adapted_learning_rate = step_size * bias_correction1 * rsqrtf(bias_correction2);


      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols)
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

           sw[i] -= adapted_learning_rate * sm[i] / (sqrtf(sv[i]) + eps);
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
        }
      }
    }
  }
}

  __global__ void IndexSelect(float* __restrict__ result, float* __restrict__ src, float* __restrict__ indice, int rows, int cols, int isAdd)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      const int srcIdx = indice[j];
      if (srcIdx >= 0)
      {
          float* resultRow = result + j * cols;
          float* srcRow = src + srcIdx * cols;

          for(int tid = 0; tid < cols; tid += blockDim.x) {
            int id = tid + threadIdx.x;
            if(id < cols) {

            if (isAdd == 0)
            {
               resultRow[id] = srcRow[id];
            }
            else
            {             
               atomicAdd(resultRow + id, srcRow[id]);
            }
            }
          }
      }
    }
  }
}

  __global__ void RoPE(float* __restrict__ result, float* __restrict__ src, int rows, int cols, int seqLen)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x)
    {
    int j = bid + blockIdx.x;
    if(j < rows)
    {
      float* resultRow = result + j * cols;
      float* srcRow = src + j * cols;
      int m = j % seqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x)
      {
        int id = tid + threadIdx.x;
        if(id < cols)
        {          
             int i = id / 2;
             float theta = __powf(500000.0, -2.0 * i / cols);
             float theta_m = theta * m;
             float cos_theta_m = __cosf(theta_m);
             float sin_theta_m = __sinf(theta_m);

             if (id % 2 == 0)
             {
                  resultRow[id] = srcRow[id] * cos_theta_m - srcRow[id + 1] * sin_theta_m;
             }
             else
             {
                  resultRow[id] = srcRow[id] * cos_theta_m + srcRow[id - 1] * sin_theta_m;
             }
        }
      }      
    }
  }
}

  __global__ void RoPEGrad(float* __restrict__ grad, float* __restrict__ adj, int rows, int cols, int seqLen)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x)
    {
    int j = bid + blockIdx.x;
    if(j < rows)
    {
      float* gradRow = grad + j * cols;
      float* adjRow = adj + j * cols;
      int m = j % seqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x)
      {
        int id = tid + threadIdx.x;
        if(id < cols)
        {          
             int i = id / 2;
             float theta = __powf(500000.0, -2.0 * i / cols);
             float theta_m = theta * m;
             float cos_theta_m = __cosf(theta_m);
             float sin_theta_m = __sinf(theta_m);

             if (id % 2 == 0)
             {
                  gradRow[id] += (adjRow[id] * cos_theta_m + adjRow[id + 1] * sin_theta_m);
             }
             else
             {
                  gradRow[id] += (adjRow[id] * cos_theta_m - adjRow[id - 1] * sin_theta_m);             
             }
        }
      }      
    }
  }
}

  

  __global__ void BuildSrcTgtMask(float* __restrict__ result, float* __restrict__ srcOriginalLengths, float* __restrict__ tgtOriginalLengths, int rows, int cols, int tgtPaddedSeqLen, float value, float maskedValue)
{

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      float* resultRow = result + j * cols;
      int batchIdx = j / tgtPaddedSeqLen;
      int seqIdxInBatch = j % tgtPaddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int srcOriginalLength = srcOriginalLengths[batchIdx];
         int tgtOriginalLength = tgtOriginalLengths[batchIdx];


         if (id < srcOriginalLength && seqIdxInBatch < tgtOriginalLength)
         {
             resultRow[id] = value;
         }
         else
         {
             resultRow[id] = maskedValue;
         }

        }
      }
    }
  }
}

  __global__ void BuildSelfMask(float* __restrict__ result, float* __restrict__ originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
{

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      float* resultRow = result + j * cols;
      int batchIdx = j / paddedSeqLen;
      int seqIdxInBatch = j % paddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int originalLength = originalLengths[batchIdx];

         if (id < originalLength && seqIdxInBatch < originalLength)
         {
             resultRow[id] = value;
         }
         else
         {
             resultRow[id] = maskedValue;
         }

        }
      }
    }
  }
}

  __global__ void BuildSelfTriMask(float* __restrict__ result, float* __restrict__ originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
{

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      float* resultRow = result + j * cols;
      int batchIdx = j / paddedSeqLen;
      int seqIdxInBatch = j % paddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int originalLength = originalLengths[batchIdx];

         if (id < originalLength && seqIdxInBatch < originalLength && id <= seqIdxInBatch)
         {
             resultRow[id] = value;
         }
         else
         {
             resultRow[id] = maskedValue;
         }

        }
      }
    }
  }
}

  __global__ void BuildTriMask(float* __restrict__ result, int rows, int cols, float value, float maskedValue)
{

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* resultRow = result + j * cols;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

            if (id <= j)
            {
                resultRow[id] = value;
            }
            else
            {
                resultRow[id] = maskedValue;
            }
        }
      }
    }
  }
}

  __global__ void IndexSelectGrad(float* __restrict__ grad, float* __restrict__ adj, float* __restrict__ indice, int rows, int cols)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      const int gradIdx = indice[j];
      if (gradIdx >= 0)
      {
          float* adjRow = adj + j * cols;
          float* gradRow = grad + gradIdx * cols;

          for(int tid = 0; tid < cols; tid += blockDim.x) {
            int id = tid + threadIdx.x;
            if(id < cols) {
            atomicAdd(gradRow + id, adjRow[id]);
            }
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

__global__ void IsCorrupted(float *in, unsigned rows, unsigned cols, int *result)
{
for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols) {
           if (!isfinite(sp[i]))
           {
             *result = 1;
             return;
           }
        }
      }      
    }
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
      _max[threadIdx.x] = -1.70141e+38;
      
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

__device__ void swap(float *a, float *b) {
	const float t = *a;
	*a = *b;
	*b = t;
}

__device__ void maxHeapify(float *maxHeap, float *maxHeapIdx, int heapSize, int idx) {

    while (1)
{
	int largest = idx;  // Initialize largest as root
	int left = (idx << 1) + 1;  // left = 2*idx + 1
	int right = (idx + 1) << 1; // right = 2*idx + 2

	// See if left child of root exists and is greater than root
	if (left < heapSize && maxHeap[left] < maxHeap[largest]) {
		largest = left;
	}

	// See if right child of root exists and is greater than
	// the largest so far
	if (right < heapSize && maxHeap[right] < maxHeap[largest]) {
		largest = right;
	}

	// Change root, if needed
	if (largest != idx) {
		swap(&maxHeap[largest], &maxHeap[idx]);
        swap(&maxHeapIdx[largest], &maxHeapIdx[idx]);
	//	maxHeapify(maxHeap, maxHeapIdx, heapSize, largest);

        idx = largest;

	}
    else
    {
       break;
    }
}
}

// A utility function to create a max heap of given capacity
__device__ void createAndBuildHeap(float *array, float* arrayIdx, int size) {
	// Start from bottommost and rightmost internal mode and heapify all
	// internal modes in bottom up way
	for (int i = (size - 2) / 2; i >= 0; --i) {
		maxHeapify(array, arrayIdx, size, i);
	}
}


__global__ void TopK(float* input, float* output, float *outputIdx, int k, unsigned rows, unsigned cols)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      float* outputRow = output + j * k;
      float* outputIdxRow = outputIdx + j * k;
      const float* inputRow = input + j * cols;

      
	if (threadIdx.x == 0) {

        for (int i = 0;i < k;i++)
        {
           outputRow[i] = inputRow[i];
           outputIdxRow[i] = i;
        }

		// Build a heap from the input data.
		createAndBuildHeap(outputRow, outputIdxRow, k);

        for (int i = k;i < cols;i++)
        {
            if (inputRow[i] > outputRow[0])
            {
               outputRow[0] = inputRow[i];
               outputIdxRow[0] = i;

               maxHeapify(outputRow, outputIdxRow, k, 0);
            }
        }
	}

  }
}
}
}
";

        private static readonly string CodeHalf = @"
#include <cuda_fp16.h>
extern ""C""
{

__global__ void gLNormalizationHalf(__half* out,
                                const __half* in,
                                const __half* alpha,
                                const __half* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      __half* so = out + j * cols;
      const __half* sp = in + j * cols;

      float* _sum = _share;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += __half2float(sp[id]);
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
          float ex = __half2float(sp[id]) - mean;
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
          float t = __half2float(alpha[id]) * (__half2float(sp[id]) - mean) / sigma;
          if(beta)
            t += __half2float(beta[id]);
          so[id] = __float2half(t);
        }
      }
    }
    __syncthreads();
  }
}

__global__ void gLayerNormalizationGradHalf(__half* gradX,
                                        __half* gradGamma,
                                        __half* gradBeta,
                                        __half* adj,
                                        __half* y,
                                        __half* x,
                                        __half* gamma,
                                        __half* beta,
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

      const __half* xRow = x + j * cols;
      const __half* yRow = y + j * cols;
      const __half* adjRow = adj + j * cols;
      __half* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += __half2float(xRow[id]);
          sum_adj_x[threadIdx.x]
              += __half2float(adjRow[id]) * (__half2float(yRow[id]) - ((beta) ? __half2float(beta[id]) : 0)) / __half2float(gamma[id]);
          sum_adj[threadIdx.x] += __half2float(adjRow[id]);
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
          float ex = __half2float(xRow[id]) - mean;
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
          float x_hat = (__half2float(yRow[id]) - ((beta) ? __half2float(beta[id]) : 0)) / __half2float(gamma[id]);
          grad_x += cols * __half2float(adjRow[id]);
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = __half2float(gamma[id]) * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000.0f ? sign * 1000.0f : valX;

          gradXRow[id] = __hadd(gradXRow[id], __float2half(valX));
          atomicAdd(gradGamma + id, __float2half(__half2float(adjRow[id]) * x_hat));
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
    __syncthreads();
  }
}



__global__ void RMSNormHalf(__half* out,
                                const __half* in,
                                const __half* alpha,
                                const __half* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      __half* so = out + j * cols;
      const __half* sp = in + j * cols;

      float* _sqSum = _share;
      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = __half2float(sp[id]);
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      int len = blockDim.x;
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
          float t = __half2float(alpha[id]) * __half2float(sp[id]) / sigma;
          if(beta)
            t += __half2float(beta[id]);
          so[id] = __float2half(t);
        }
      }
    }
    __syncthreads();
  }
}



__global__ void RMSNormGradHalf(__half* gradX,
                                        __half* gradGamma,
                                        __half* gradBeta,
                                        __half* adj,
                                        __half* y,
                                        __half* x,
                                        __half* gamma,
                                        __half* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_sqr = shared + 2 * blockDim.x;

      const __half* xRow = x + j * cols;
      const __half* yRow = y + j * cols;
      const __half* adjRow = adj + j * cols;
      __half* gradXRow = gradX + j * cols;

      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_adj_x[threadIdx.x]
              += __half2float(adjRow[id]) * (__half2float(yRow[id]) - ((beta) ? __half2float(beta[id]) : 0)) / __half2float(gamma[id]);
          sum_adj[threadIdx.x] += __half2float(adjRow[id]);
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = __half2float(xRow[id]);
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
          float x_hat = (__half2float(yRow[id]) - ((beta) ? __half2float(beta[id]) : 0)) / __half2float(gamma[id]);
          grad_x += cols * __half2float(adjRow[id]);
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          float valX = __half2float(gamma[id]) * grad_x;
          float sign = (0.f < valX) - (valX < 0.f);
          valX = fabs(valX) > 1000.0f ? sign * 1000.0f : valX;

          gradXRow[id] = __hadd(gradXRow[id], __float2half(valX));
          atomicAdd(gradGamma + id, __float2half(__half2float(adjRow[id]) * x_hat));
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
    __syncthreads();
  }
}



__global__ void RoPEGradHalf(__half* __restrict__ grad, __half* __restrict__ adj, int rows, int cols, int seqLen)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x)
    {
    int j = bid + blockIdx.x;
    if(j < rows)
    {
      __half* gradRow = grad + j * cols;
      __half* adjRow = adj + j * cols;
      int m = j % seqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x)
      {
        int id = tid + threadIdx.x;
        if(id < cols)
        {          
             int i = id / 2;
             float theta = __powf(500000.0, -2.0 * i / cols);
             float theta_m = theta * m;
             float cos_theta_m = __cosf(theta_m);
             float sin_theta_m = __sinf(theta_m);

             if (id % 2 == 0)
             {
                  gradRow[id] = __float2half(__half2float(gradRow[id]) + __half2float(adjRow[id]) * cos_theta_m + __half2float(adjRow[id + 1]) * sin_theta_m);
             }
             else
             {
                  gradRow[id] = __float2half(__half2float(gradRow[id]) + __half2float(adjRow[id]) * cos_theta_m - __half2float(adjRow[id - 1]) * sin_theta_m);             
             }
        }
      }      
    }
  }
}



  __global__ void RoPEHalf(__half* __restrict__ result, __half* __restrict__ src, int rows, int cols, int seqLen)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x)
    {
    int j = bid + blockIdx.x;
    if(j < rows)
    {
      __half* resultRow = result + j * cols;
      __half* srcRow = src + j * cols;
      int m = j % seqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x)
      {
        int id = tid + threadIdx.x;
        if(id < cols)
        {          
             int i = id / 2;
             float theta = __powf(500000.0, -2.0 * i / cols);
             float theta_m = theta * m;
             float cos_theta_m = __cosf(theta_m);
             float sin_theta_m = __sinf(theta_m);

             if (id % 2 == 0)
             {
                  resultRow[id] = __float2half(__half2float(srcRow[id]) * cos_theta_m - __half2float(srcRow[id + 1]) * sin_theta_m);
             }
             else
             {
                  resultRow[id] = __float2half(__half2float(srcRow[id]) * cos_theta_m + __half2float(srcRow[id - 1]) * sin_theta_m);
             }
        }
      }      
    }
  }
}

__global__ void AdamHalf(__half* __restrict__ w, __half* __restrict__ g, float* __restrict__ v, float* __restrict__ m, unsigned rows, unsigned cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
{
  for(int bid = 0; bid < rows; bid += gridDim.x) 
  {
    int j = bid + blockIdx.x;
    if(j < rows) 
    {
      __half* sw = w + j * cols;
      __half* sg = g + j * cols;
      float* sv = v + j * cols;
      float* sm = m + j * cols;

      float bias_correction1 = 1.0 / (1.0 - powf(decay_rate_m, iter));
      float bias_correction2 = 1.0 / (1.0 - powf(decay_rate_v, iter));
      float adapted_learning_rate = step_size * bias_correction1 * rsqrtf(bias_correction2);


      for(int tid = 0; tid < cols; tid += blockDim.x) 
      {        
        int i = tid + threadIdx.x;
        if(i < cols)
        {
           float g = __half2float(sg[i]) / batchSize;

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

           sw[i] = __float2half(__half2float(sw[i]) - (adapted_learning_rate * sm[i] / (sqrtf(sv[i]) + eps)));
        }
      }
    }
  }
}

 __global__ void IndexSelectHalf(__half* __restrict__ result, __half* __restrict__ src, float* __restrict__ indice, int rows, int cols, int isAdd)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      const int srcIdx = indice[j];
      if (srcIdx >= 0)
      {
          __half* resultRow = result + j * cols;
          __half* srcRow = src + srcIdx * cols;

          for(int tid = 0; tid < cols; tid += blockDim.x) {
            int id = tid + threadIdx.x;
            if(id < cols) {

            if (isAdd == 0)
            {
               resultRow[id] = srcRow[id];
            }
            else
            {             
               atomicAdd(resultRow + id, srcRow[id]);
            }
            }
          }
      }
    }
  }
}


  __global__ void BuildSrcTgtMaskHalf(__half* __restrict__ result, float* __restrict__ srcOriginalLengths, float* __restrict__ tgtOriginalLengths, int rows, int cols, int tgtPaddedSeqLen, float value, float maskedValue)
{
      __half hvalue = __float2half(value);
      __half hmaskedValue = __float2half(maskedValue);

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      __half* resultRow = result + j * cols;
      int batchIdx = j / tgtPaddedSeqLen;
      int seqIdxInBatch = j % tgtPaddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int srcOriginalLength = srcOriginalLengths[batchIdx];
         int tgtOriginalLength = tgtOriginalLengths[batchIdx];

         if (id < srcOriginalLength && seqIdxInBatch < tgtOriginalLength)
         {
             resultRow[id] = hvalue;
         }
         else
         {
             resultRow[id] = hmaskedValue;
         }

        }
      }
    }
  }
}

  __global__ void BuildSelfMaskHalf(__half* __restrict__ result, float* __restrict__ originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
{
      __half hvalue = __float2half(value);
      __half hmaskedValue = __float2half(maskedValue);

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      __half* resultRow = result + j * cols;
      int batchIdx = j / paddedSeqLen;
      int seqIdxInBatch = j % paddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int originalLength = originalLengths[batchIdx];

         if (id < originalLength && seqIdxInBatch < originalLength)
         {
             resultRow[id] = hvalue;
         }
         else
         {
             resultRow[id] = hmaskedValue;
         }

        }
      }
    }
  }
}

  __global__ void BuildSelfTriMaskHalf(__half* __restrict__ result, float* __restrict__ originalLengths, int rows, int cols, int paddedSeqLen, float value, float maskedValue)
{
      __half hvalue = __float2half(value);
      __half hmaskedValue = __float2half(maskedValue);

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      __half* resultRow = result + j * cols;
      int batchIdx = j / paddedSeqLen;
      int seqIdxInBatch = j % paddedSeqLen;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
         int originalLength = originalLengths[batchIdx];

         if (id < originalLength && seqIdxInBatch < originalLength && id <= seqIdxInBatch)
         {
             resultRow[id] = hvalue;
         }
         else
         {
             resultRow[id] = hmaskedValue;
         }

        }
      }
    }
  }
}
  __global__ void BuildTriMaskHalf(__half* __restrict__ result, int rows, int cols, float value, float maskedValue)
{
      __half hvalue = __float2half(value);
      __half hmaskedValue = __float2half(maskedValue);

    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      __half* resultRow = result + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

            if (id <= j)
            {
                resultRow[id] = hvalue;
            }
            else
            {
                resultRow[id] = hmaskedValue;
            }
        }
      }
    }
  }
}

 __global__ void IndexSelectGradHalf(__half* __restrict__ grad, __half* __restrict__ adj, float* __restrict__ indice, int rows, int cols)
  {
    for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      const int gradIdx = indice[j];
      if (gradIdx >= 0)
      {
          __half* adjRow = adj + j * cols;
          __half* gradRow = grad + gradIdx * cols;

          for(int tid = 0; tid < cols; tid += blockDim.x) {
            int id = tid + threadIdx.x;
            if(id < cols) {
            atomicAdd(gradRow + id, adjRow[id]);
            }
          }
      }
    }
  }
}

   __global__ void gSoftmaxGradHalf(__half* grad, __half* adj, __half* val, int rows, int cols, int addGrad)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share;

      __half* gradRow = grad + j * cols;
      const __half* adjRow = adj + j * cols;
      const __half* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float v = __half2float(valRow[id]) * __half2float(adjRow[id]);
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

         float v = __half2float(valRow[id]) * __half2float(adjRow[id]);

         if (addGrad == 0)
         {
             gradRow[id] = __float2half(v - sum * __half2float(valRow[id]));
         }
         else
         {
             gradRow[id] = __float2half((v + __half2float(gradRow[id])) - sum * __half2float(valRow[id]));
         }

         
        }
      }
    }
    __syncthreads();
  }
}

__global__ void IsCorruptedHalf(__half *in, unsigned rows, unsigned cols, int *result)
{
for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const __half* sp = in + j * cols;      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols) {
           if (!isfinite(__half2float(sp[i])))
           {
             *result = 1;
             return;
           }
        }
      }      
    }
  }
}


 __global__ void gSoftmaxHalf(__half* out, __half* in, unsigned rows, unsigned cols)
  {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      __half* so = out + j * cols;
      const __half* sp = in + j * cols;

      extern __shared__ float _share[];     
      float* _max = _share;
      _max[threadIdx.x] = -1.70141e+38;
      
      for(int tid = 0; tid < cols; tid += blockDim.x) {        
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(__half2float(sp[i]) > _max[threadIdx.x])
            _max[threadIdx.x] = __half2float(sp[i]);
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
          float ex = expf(__half2float(sp[i]) - max);
          so[i] = __float2half(ex);
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
          so[i] = __float2half(__half2float(so[i]) / sum);
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
            if (TSCudaContext.ElementType == DType.Float16)
            {
                Logger.WriteLine(Logger.Level.debug, "Building advanced kernels for both FP32 and FP16.");

                return Code + CodeHalf;
            }
            else
            {
                Logger.WriteLine(Logger.Level.debug, "Building advanced kernels for both FP32.");

                return Code;
            }
        }



        public Tensor TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(inVal);
            TopK(context, outVal, outIdx, inVal, k);

            return outVal;
        }


        private void TopK(TSCudaContext context, Tensor outVal, Tensor outIdx, Tensor inVal, int k)
        {
            CudaContext cudaContext = context.CudaContextForTensor(inVal);

            cudaContext.SetCurrent();

            int ndim = inVal.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(inVal.Sizes, inVal.Strides);
            long cols = inVal.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr outValPtr = CudaHelpers.GetBufferStart(outVal);
            CUdeviceptr outIdxPtr = CudaHelpers.GetBufferStart(outIdx);
            CUdeviceptr inValPtr = CudaHelpers.GetBufferStart(inVal);


            Invoke(context, cudaContext, "TopK", grid, block, (uint)(block.x * sizeof(float) * 2), CUstream.NullStream, inValPtr, outValPtr, outIdxPtr, k, rows, cols);

        }


        public Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(inGrad);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(outGrad, inGrad, false, inGrad.Sizes);
            LayerNormGrad(context, writeTarget, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNormGrad(TSCudaContext context, Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();

            int ndim = inGrad.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(inGrad.Sizes, inGrad.Strides);
            long cols = inGrad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr outGradPtr = CudaHelpers.GetBufferStart(outGrad);
            CUdeviceptr alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            CUdeviceptr betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            CUdeviceptr inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            CUdeviceptr yPtr = CudaHelpers.GetBufferStart(y);
            CUdeviceptr xPtr = CudaHelpers.GetBufferStart(x);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);

            string kernelName = "gLayerNormalizationGrad";
            if (outGrad.ElementType == DType.Float16)
            {
                kernelName = "gLayerNormalizationGradHalf";
            }
            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float) * 4, CUstream.NullStream, outGradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, xPtr, alphaPtr, betaPtr, rows, cols, eps);
        }


        public Tensor RMSNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(inGrad);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(outGrad, inGrad, false, inGrad.Sizes);
            RMSNormGrad(context, writeTarget, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps);

            return writeTarget;
        }


        private void RMSNormGrad(TSCudaContext context, Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();

            int ndim = inGrad.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(inGrad.Sizes, inGrad.Strides);
            long cols = inGrad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr outGradPtr = CudaHelpers.GetBufferStart(outGrad);
            CUdeviceptr alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            CUdeviceptr betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            CUdeviceptr inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            CUdeviceptr yPtr = CudaHelpers.GetBufferStart(y);
            CUdeviceptr xPtr = CudaHelpers.GetBufferStart(x);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);

            string kernelName = "RMSNormGrad";
            if (outGrad.ElementType == DType.Float16)
            {
                kernelName = "RMSNormGradHalf";
            }
            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float) * 4, CUstream.NullStream, outGradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, xPtr, alphaPtr, betaPtr, rows, cols, eps);
        }

        public void AddLayerNormGrad(Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(inGrad);
            Tensor writeTarget1 = TensorResultBuilder.GetWriteTarget(out1Grad, inGrad, false, inGrad.Sizes);
            Tensor writeTarget2 = TensorResultBuilder.GetWriteTarget(out2Grad, inGrad, false, inGrad.Sizes);
            AddLayerNormGrad(context, writeTarget1, writeTarget2, alphaGrad, betaGrad, inGrad, y, x1, x2, alpha, beta, eps);
        }

        private void AddLayerNormGrad(TSCudaContext context, Tensor out1Grad, Tensor out2Grad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x1, Tensor x2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(inGrad);

            cudaContext.SetCurrent();
            int ndim = inGrad.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(inGrad.Sizes, inGrad.Strides);
            long cols = inGrad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr out1GradPtr = CudaHelpers.GetBufferStart(out1Grad);
            CUdeviceptr out2GradPtr = CudaHelpers.GetBufferStart(out2Grad);
            CUdeviceptr alphaGradPtr = CudaHelpers.GetBufferStart(alphaGrad);
            CUdeviceptr betaGradPtr = CudaHelpers.GetBufferStart(betaGrad);
            CUdeviceptr inGradPtr = CudaHelpers.GetBufferStart(inGrad);
            CUdeviceptr yPtr = CudaHelpers.GetBufferStart(y);
            CUdeviceptr x1Ptr = CudaHelpers.GetBufferStart(x1);
            CUdeviceptr x2Ptr = CudaHelpers.GetBufferStart(x2);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gAddLayerNormalizationGrad", grid, block, block.x * sizeof(float) * 4, CUstream.NullStream, out1GradPtr, out2GradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, x1Ptr, x2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }


        public Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            LayerNorm(context, writeTarget, src, alpha, beta, eps);

            return writeTarget;
        }


        private void LayerNorm(TSCudaContext context, Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);

            string kernelName = "gLNormalization";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "gLNormalizationHalf";
            }

            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, alphaPtr, betaPtr, rows, cols, eps);

        }


        public Tensor RMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            RMSNorm(context, writeTarget, src, alpha, beta, eps);

            return writeTarget;
        }


        private void RMSNorm(TSCudaContext context, Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);

            string kernelName = "RMSNorm";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "RMSNormHalf";
            }

            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, alphaPtr, betaPtr, rows, cols, eps);

        }

        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src1);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src1, false, src1.Sizes);
            AddLayerNorm(context, writeTarget, src1, src2, alpha, beta, eps);

            return writeTarget;
        }

        private void AddLayerNorm(TSCudaContext context, Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps = 1e-9f)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src1);

            cudaContext.SetCurrent();

            int ndim = src1.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src1.Sizes, src1.Strides);
            long cols = src1.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr src1Ptr = CudaHelpers.GetBufferStart(src1);
            CUdeviceptr src2Ptr = CudaHelpers.GetBufferStart(src2);
            CUdeviceptr alphaPtr = CudaHelpers.GetBufferStart(alpha);
            CUdeviceptr betaPtr = CudaHelpers.GetBufferStart(beta);


            Invoke(context, cudaContext, "gAddLNormalization", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, src1Ptr, src2Ptr, alphaPtr, betaPtr, rows, cols, eps);

        }



        private void BuildSrcTgtMask(TSCudaContext context, Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int tgtPaddedSeqLen, float value, float maskedValue)
        {
            CudaContext cudaContext = context.CudaContextForTensor(srcOriginalLengths);

            cudaContext.SetCurrent();

            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcOriginalLengthsPtr = CudaHelpers.GetBufferStart(srcOriginalLengths);
            CUdeviceptr tgtOriginalLengthsPtr = CudaHelpers.GetBufferStart(tgtOriginalLengths);


            string kernelName = "BuildSrcTgtMask";
            if (result.ElementType == DType.Float16)
            {
                kernelName = "BuildSrcTgtMaskHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcOriginalLengthsPtr, tgtOriginalLengthsPtr, rows, cols, tgtPaddedSeqLen, value, maskedValue);
        }



        private void BuildSelfMask(TSCudaContext context, Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            CudaContext cudaContext = context.CudaContextForTensor(originalLengths);

            cudaContext.SetCurrent();

            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr originalLengthsPtr = CudaHelpers.GetBufferStart(originalLengths);

            string kernelName = "BuildSelfMask";
            if (result.ElementType == DType.Float16)
            {
                kernelName = "BuildSelfMaskHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, originalLengthsPtr, rows, cols, paddedSeqLen, value, maskedValue);


        }


        private void BuildSelfTriMask(TSCudaContext context, Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            CudaContext cudaContext = context.CudaContextForTensor(originalLengths);

            cudaContext.SetCurrent();

            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr originalLengthsPtr = CudaHelpers.GetBufferStart(originalLengths);

            string kernelName = "BuildSelfTriMask";
            if (result.ElementType == DType.Float16)
            {
                kernelName = "BuildSelfTriMaskHalf";
            }

            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, originalLengthsPtr, rows, cols, paddedSeqLen, value, maskedValue);
        }



        private void BuildTriMask(TSCudaContext context, Tensor result, float value, float maskedValue)
        {
            CudaContext cudaContext = context.CudaContextForTensor(result);

            cudaContext.SetCurrent();

            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);

            string kernelName = "BuildTriMask";
            if (result.ElementType == DType.Float16)
            {
                kernelName = "BuildTriMaskHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, rows, cols, value, maskedValue);
        }

        private void IndexSelect(TSCudaContext context, Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            if (result.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(result)} is not contiguous.");
            }
            if (src.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(src)} is not contiguous.");
            }
            if (indice.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(indice)} is not contiguous.");
            }


            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);
            CUdeviceptr indicePtr = CudaHelpers.GetBufferStart(indice);

            string kernelName = "IndexSelect";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "IndexSelectHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, indicePtr, rows, cols, (isAdd == true) ? 1 : 0);

        }


        private void RoPE(TSCudaContext context, Tensor result, Tensor src, int seqLen)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            if (src.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(src)} is not contiguous.");
            }
            if (result.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(result)} is not contiguous.");
            }


            int ndim = result.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(result.Sizes, result.Strides);
            long cols = result.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);

            string kernelName = "RoPE";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "RoPEHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, rows, cols, seqLen);

        }

        private void RoPEGrad(TSCudaContext context, Tensor grad, Tensor adj, int seqLen)
        {
            CudaContext cudaContext = context.CudaContextForTensor(adj);

            cudaContext.SetCurrent();

            if (grad.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(grad)} is not contiguous.");
            }
            if (adj.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(adj)} is not contiguous.");
            }


            int ndim = adj.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(adj.Sizes, adj.Strides);
            long cols = adj.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr gradPtr = CudaHelpers.GetBufferStart(grad);
            CUdeviceptr adjPtr = CudaHelpers.GetBufferStart(adj);

            string kernelName = "RoPEGrad";
            if (adj.ElementType == DType.Float16)
            {
                kernelName = "RoPEGradHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, rows, cols, seqLen);
        }


        private void IndexSelectGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor indice)
        {
            CudaContext cudaContext = context.CudaContextForTensor(adj);

            cudaContext.SetCurrent();

            if (grad.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(grad)} is not contiguous.");
            }
            if (adj.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(adj)} is not contiguous.");
            }
            if (indice.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(indice)} is not contiguous.");
            }


            int ndim = adj.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(adj.Sizes, adj.Strides);
            long cols = adj.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr gradPtr = CudaHelpers.GetBufferStart(grad);
            CUdeviceptr adjPtr = CudaHelpers.GetBufferStart(adj);
            CUdeviceptr indicePtr = CudaHelpers.GetBufferStart(indice);

            string kernelName = "IndexSelectGrad";
            if (adj.ElementType == DType.Float16)
            {
                kernelName = "IndexSelectGradHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, indicePtr, rows, cols);

        }


        private bool IsCorrupted(TSCudaContext context, Tensor src)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);
            cudaContext.SetCurrent();

            if (src.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(src)} is not contiguous.");
            }

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            int[] rets = new int[1];
            rets[0] = 0;
            Tensor result = new Tensor(src.Allocator, DType.Int32, sizes: new long[] { 1, 1 });
            result.SetElementsAsInt(rets);

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);

            string kernelName = "IsCorrupted";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "IsCorruptedHalf";
            }

            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, srcPtr, rows, cols, resultPtr);

            rets = result.GetElementsAsInt(1);
            if (rets[0] == 0)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        private void Softmax(TSCudaContext context, Tensor result, Tensor src)
        {
            if (result.ElementType != src.ElementType)
            {
                throw new ArgumentException($"The element type between source and result must be same.");
            }

            CudaContext cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            if (result.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(result)} is not contiguous.");
            }
            if (src.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(src)} is not contiguous.");
            }

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;


            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr resultPtr = CudaHelpers.GetBufferStart(result);
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);

            string kernelName = "gSoftmax";
            if (src.ElementType == DType.Float16)
            {
                kernelName = "gSoftmaxHalf";
            }

            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, rows, cols);
        }

        private void Adam(TSCudaContext context, Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            //if (weight.ElementType != gradient.ElementType)
            //{
            //    throw new ArgumentException($"The element type between weights and gradients must be same.");
            //}

            CudaContext cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            int ndim = weight.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(weight.Sizes, weight.Strides);
            long cols = weight.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr weightPtr = CudaHelpers.GetBufferStart(weight);
            CUdeviceptr gradientPtr = CudaHelpers.GetBufferStart(gradient);
            CUdeviceptr vPtr = CudaHelpers.GetBufferStart(v);
            CUdeviceptr mPtr = CudaHelpers.GetBufferStart(m);

            string kernelName = "Adam";
            if (weight.ElementType == DType.Float16)
            {
                kernelName = "AdamHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, weightPtr, gradientPtr, vPtr, mPtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
        }

        public Tensor Adam(Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(weight);
            Adam(context, weight, gradient, v, m, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);

            return weight;
        }

        private void RMSProp(TSCudaContext context, Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            CudaContext cudaContext = context.CudaContextForTensor(weight);

            cudaContext.SetCurrent();

            int ndim = weight.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(weight.Sizes, weight.Strides);
            long cols = weight.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr weightPtr = CudaHelpers.GetBufferStart(weight);
            CUdeviceptr gradientPtr = CudaHelpers.GetBufferStart(gradient);
            CUdeviceptr cachePtr = CudaHelpers.GetBufferStart(cache);

            Invoke(context, cudaContext, "RMSProp", grid, block, 0, CUstream.NullStream, weightPtr, gradientPtr, cachePtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate, eps);
        }

        public Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(weight);
            RMSProp(context, weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);

            return weight;
        }

        private void SoftmaxGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            CudaContext cudaContext = context.CudaContextForTensor(grad);

            cudaContext.SetCurrent();

            if (grad.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(grad)} is not contiguous.");
            }
            if (adj.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(adj)} is not contiguous.");
            }
            if (val.IsContiguous() == false)
            {
                throw new Exception($"Tensor {nameof(val)} is not contiguous.");
            }


            int ndim = grad.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(grad.Sizes, grad.Strides);
            long cols = grad.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            int iAddGrad = addGrad ? 1 : 0;

            dim3 block = new dim3((uint)Math.Min(512, cols));
            dim3 grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(rows, block.y)));

            CUdeviceptr gradPtr = CudaHelpers.GetBufferStart(grad);
            CUdeviceptr adjPtr = CudaHelpers.GetBufferStart(adj);
            CUdeviceptr valPtr = CudaHelpers.GetBufferStart(val);

            string kernelName = "gSoftmaxGrad";
            if (val.ElementType == DType.Float16)
            {
                kernelName = "gSoftmaxGradHalf";
            }


            Invoke(context, cudaContext, kernelName, grid, block, block.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, valPtr, rows, cols, iAddGrad);
        }

        public bool IsCorrupted(Tensor src)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            return IsCorrupted(context, src);
        }

        public Tensor Softmax(Tensor result, Tensor src)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            Softmax(context, writeTarget, src);

            return writeTarget;
        }

        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(grad);
            SoftmaxGrad(context, grad, adj, val, addGrad);

            return grad;
        }


        public Tensor BuildSrcTgtMask(Tensor result, Tensor srcOriginalLengths, Tensor tgtOriginalLengths, int srcPaddedSeqLen, int tgtPaddedSeqLen, float value, float maskedValue)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(srcOriginalLengths);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcOriginalLengths, true, new long[] { srcOriginalLengths.Sizes[0], tgtPaddedSeqLen, srcPaddedSeqLen });

            BuildSrcTgtMask(context, writeTarget, srcOriginalLengths, tgtOriginalLengths, tgtPaddedSeqLen, value, maskedValue);

            return writeTarget;
        }


        public Tensor BuildSelfMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(originalLengths);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, originalLengths, true, new long[] { originalLengths.Sizes[0], paddedSeqLen, paddedSeqLen });

            BuildSelfMask(context, writeTarget, originalLengths, paddedSeqLen, value, maskedValue);

            return writeTarget;
        }


        public Tensor BuildSelfTriMask(Tensor result, Tensor originalLengths, int paddedSeqLen, float value, float maskedValue)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(originalLengths);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, originalLengths, true, new long[] { originalLengths.Sizes[0], paddedSeqLen, paddedSeqLen });

            BuildSelfTriMask(context, writeTarget, originalLengths, paddedSeqLen, value, maskedValue);

            return writeTarget;
        }

        public Tensor BuildTriMask(Tensor result, float value, float maskedValue)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(result);
            BuildTriMask(context, result, value, maskedValue);

            return result;
        }

        public Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, new long[] { indice.Sizes[0], src.Sizes[1] });
            IndexSelect(context, writeTarget, src, indice, isAdd);

            return writeTarget;
        }

        public Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            if (grad == null)
            {
                throw new ArgumentNullException($"Tensor grad should not be null.");
            }

            TSCudaContext context = CudaHelpers.TSContextForTensor(adj);
            IndexSelectGrad(context, grad, adj, indice);

            return grad;
        }


        public Tensor RoPE(Tensor result, Tensor src, int seqLen)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(src);
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            RoPE(context, writeTarget, src, seqLen);

            return writeTarget;
        }

        public Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen)
        {
            TSCudaContext context = CudaHelpers.TSContextForTensor(adj);
            RoPEGrad(context, grad, adj, seqLen);

            return grad;
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
