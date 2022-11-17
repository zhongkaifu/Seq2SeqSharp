// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
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

        public AdvFuncKernels()
            : base(GetFullCode())
        {
        }

        private static string GetFullCode()
        {
            return Code;
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


            Invoke(context, cudaContext, "gLayerNormalizationGrad", grid, block, block.x * sizeof(float) * 4, CUstream.NullStream, outGradPtr, alphaGradPtr, betaGradPtr, inGradPtr, yPtr, xPtr, alphaPtr, betaPtr, rows, cols, eps);

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


            Invoke(context, cudaContext, "gLNormalization", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, alphaPtr, betaPtr, rows, cols, eps);

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

            Invoke(context, cudaContext, "BuildSrcTgtMask", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcOriginalLengthsPtr, tgtOriginalLengthsPtr, rows, cols, tgtPaddedSeqLen, value, maskedValue);
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


            Invoke(context, cudaContext, "BuildSelfMask", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, originalLengthsPtr, rows, cols, paddedSeqLen, value, maskedValue);


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


            Invoke(context, cudaContext, "BuildSelfTriMask", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, originalLengthsPtr, rows, cols, paddedSeqLen, value, maskedValue);
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


            Invoke(context, cudaContext, "BuildTriMask", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, rows, cols, value, maskedValue);
        }



        private void IndexSelect(TSCudaContext context, Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            CudaContext cudaContext = context.CudaContextForTensor(src);

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
            CUdeviceptr srcPtr = CudaHelpers.GetBufferStart(src);
            CUdeviceptr indicePtr = CudaHelpers.GetBufferStart(indice);


            Invoke(context, cudaContext, "IndexSelect", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, indicePtr, rows, cols, (isAdd == true) ? 1 : 0);

        }


        private void IndexSelectGrad(TSCudaContext context, Tensor grad, Tensor adj, Tensor indice)
        {
            CudaContext cudaContext = context.CudaContextForTensor(adj);

            cudaContext.SetCurrent();

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


            Invoke(context, cudaContext, "IndexSelectGrad", grid, block, block.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, indicePtr, rows, cols);

        }



        private void Softmax(TSCudaContext context, Tensor result, Tensor src)
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

            Invoke(context, cudaContext, "gSoftmax", grid, block, block.x * sizeof(float), CUstream.NullStream, resultPtr, srcPtr, rows, cols);
        }

        private void Adam(TSCudaContext context, Tensor weight, Tensor gradient, Tensor v, Tensor m, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
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
            CUdeviceptr vPtr = CudaHelpers.GetBufferStart(v);
            CUdeviceptr mPtr = CudaHelpers.GetBufferStart(m);

            Invoke(context, cudaContext, "Adam", grid, block, 0, CUstream.NullStream, weightPtr, gradientPtr, vPtr, mPtr, rows, cols, batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
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

            Invoke(context, cudaContext, "gSoftmaxGrad", grid, block, block.x * sizeof(float), CUstream.NullStream, gradPtr, adjPtr, valPtr, rows, cols, iAddGrad);
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
