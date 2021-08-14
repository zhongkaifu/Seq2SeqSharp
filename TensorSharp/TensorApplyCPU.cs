﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp
{
	public class TensorApplyCPU
	{
		unsafe public delegate void Apply1KernelFunction(float* x);
		unsafe public delegate void Apply2KernelFunction(float* x, float* y);
		unsafe public delegate void Apply3KernelFunction(float* x, float* y, float* z);
		unsafe public delegate void ApplyDim2KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY);
		unsafe public delegate void ApplyDim3KernelFuncton(float* x, long sizeX, long stridesX, float* y, long sizeY, long stridesY, float* z, long sizeZ, long stridesZ);


		unsafe static void Apply1(Tensor tensor1, Apply1KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd(); tensor1Iter.BlockStep())
				{
					func(tensor1Iter.data);
				}

			} while (tensor1Iter.NextBlock());
		}


		unsafe static void Apply2(Tensor tensor1, Tensor tensor2, Apply2KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd(); tensor1Iter.BlockStep(), tensor2Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}

		unsafe static void Apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Apply3KernelFunction func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorIterState tensor1Iter = new TensorIterState(buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides);
			TensorIterState tensor2Iter = new TensorIterState(buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides);
			TensorIterState tensor3Iter = new TensorIterState(buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides);

			do
			{
				for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd();
						tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep())
				{
					func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data);
				}

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}


		unsafe static void ApplyDim2(Tensor tensor1, Tensor tensor2, int iterationDim, ApplyDim2KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
		}




		unsafe static void ApplyDim3(Tensor tensor1, Tensor tensor2, Tensor tensor3, int iterationDim, ApplyDim3KernelFuncton func)
		{
			float* buffer1 = (float*)CpuNativeHelpers.GetBufferStart(tensor1);
			float* buffer2 = (float*)CpuNativeHelpers.GetBufferStart(tensor2);
			float* buffer3 = (float*)CpuNativeHelpers.GetBufferStart(tensor3);

			TensorDimIterState tensor1Iter = new TensorDimIterState(buffer1, tensor1.DimensionCount, tensor1.Sizes, tensor1.Strides, iterationDim);
			TensorDimIterState tensor2Iter = new TensorDimIterState(buffer2, tensor2.DimensionCount, tensor2.Sizes, tensor2.Strides, iterationDim);
			TensorDimIterState tensor3Iter = new TensorDimIterState(buffer3, tensor3.DimensionCount, tensor3.Sizes, tensor3.Strides, iterationDim);

			do
			{
				func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
					tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride,
					tensor3Iter.data, tensor3Iter.size, tensor3Iter.stride);

			} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
		}




		unsafe public static void Gather_Apply(Tensor result, Tensor src, int dim, Tensor indices)
		{
			unsafe void func(float* rData, long rSize, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{
				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= sSize) { throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize}'"); }

					*(rData + i * rStride) = sData[idx * sStride];
				}
			}

			ApplyDim3(result, src, indices, dim, func);
		}



		unsafe public static void Scatter_Apply(Tensor result, Tensor src, int dim, Tensor indices)
		{
			unsafe void func(float* rData, long rSize, long rStride,
				float* sData, long sSize, long sStride,
				float* iData, long iSize, long iStride)
			{

				for (int i = 0; i < iSize; ++i)
				{
					long idx = (long)*(iData + i * iStride);
					if (idx < 0 || idx >= rSize) { throw new IndexOutOfRangeException($"Invalid index in gather. Idx = '{idx}', sSize = '{sSize}'"); }

					rData[idx * rStride] = *(sData + i * sStride);
				}

			}

			ApplyDim3(result, src, indices, dim, func);
		}




		unsafe public static void Fill_Apply(Tensor result, float value)
		{
			unsafe void func(float* r)
			{
				*r = value;
			}

			Apply1(result, func);
		}


		unsafe public static void Copy_Apply(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = *s;
			}
			Apply2(result, src, func);
		}


		unsafe public static void Sum_Apply(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float sum = 0.0f;
				for (long i = 0; i < sSize; ++i)
				{
					sum += s[i * sStride];
				}
				*r = sum;
			}
			ApplyDim2(result, src, dimension, func);
		}


		unsafe public static void Argmax_Apply(Tensor resultIndices, Tensor src, int dimension)
		{

			unsafe void func(float* rIndVal, long rIndSize, long rIndStride,
				float* s, long sSize, long sStride)
			{
				float value = s[0];
				float index = 0;
				for (long i = 1; i < sSize; ++i)
				{
					float currentVal = s[i * sStride];
					if (currentVal > value)
					{
						value = currentVal;
						index = (float)i;
					}
				}
				*rIndVal = index;
			}

			ApplyDim2(resultIndices, src, dimension, func);
		}

		unsafe public static void Max_Apply(Tensor result, Tensor src, int dimension)
		{
			unsafe void func(float* r, long rSize, long rStride, float* s, long sSize, long sStride)
			{
				float value = s[0];
				for (long i = 1; i < sSize; ++i)
				{
					value = Math.Max(value, s[i * sStride]);
				}
				*r = value;
			}

			ApplyDim2(result, src, dimension, func);
		}


		unsafe public static void Add_Apply(Tensor result, Tensor lhs, Tensor rhs)
		{
			unsafe void func(float* r, float* left, float* right)
			{
				*r = add(*left, *right);
			}

			Apply3(result, lhs, rhs, func);
		}

		unsafe public static void Add_Apply(Tensor result, Tensor src, float value)
		{
			unsafe void func(float* r, float* s)
			{
				*r = add(*s, value);
			}

			Apply2(result, src, func);
		}



		unsafe public static void Mul_Apply(Tensor result, Tensor src, float value)
		{
			unsafe void func(float* r, float* s)
			{
				*r = mul(*s, value);
			}

			Apply2(result, src, func);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relu(float w)
		{
			if (w < 0.0f)
				return 0.0f;
			return w;

		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float relud(float w, float g)
		{
			if (w > 0.0f)
				return g;
			return 0.0f;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float add(float x, float y)
		{
			return x + y;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static float mul(float x, float y)
		{
			return x * y;
		}


		unsafe static public void Relu_Apply(Tensor result, Tensor src)
		{
			unsafe void func(float* r, float* s)
			{
				*r = relu(*s);
			};

			Apply2(result, src, func);
		}


		unsafe static public void ReluD_Apply(Tensor result, Tensor srcW, Tensor srcG)
		{
			unsafe void func(float* r, float* y, float* x)
			{
				*r = relud(*y, *x);
			}

			Apply3(result, srcW, srcG, func);
		}



		unsafe static public void Softmax(Tensor tOut, Tensor tIn, int rows, int cols)
		{
			float* pOut = (float*)CpuNativeHelpers.GetBufferStart(tOut);
			float* pIn = (float*)CpuNativeHelpers.GetBufferStart(tIn);

			for (int j = 0; j < rows; ++j)
			{
				float* so = pOut + j * cols;
				float* sp = pIn + j * cols;
				
				float max = sp[0];
				for (int i = 1; i < cols; ++i)
					max = Math.Max(max, sp[i]);

				float sum = 0.0f;
                for (int i = 0; i < cols; ++i)
                {
                    float ex = (float)Math.Exp(sp[i] - max);
                    so[i] = ex;
                    sum += ex;
                }

				//for (int i = 0; i < cols; ++i)
				//{
				//	so[i] /= sum;
				//}



				Span<float> spanSO = new Span<float>(so, cols);
				int vectorSize = Vector<float>.Count;
				int k = 0;
				Vector<float> vecSum = new Vector<float>(sum);
				for (k = 0; k < cols - vectorSize; k += vectorSize)
				{
					Vector<float> vecSO = new Vector<float>(spanSO.Slice(k));
					vecSO /= vecSum;

					vecSO.CopyTo(spanSO.Slice(k));
				}
				for (; k < cols; k++)
				{
					so[k] /= sum;
				}



			}
		}


		unsafe static public void SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_, int rows, int cols, bool addGrad)
		{

			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* val = (float*)CpuNativeHelpers.GetBufferStart(val_);

			for (int j = 0; j < rows; ++j)
			{
				float* gradRow = grad + j * cols;
				float* adjRow = adj + j * cols;
				float* valRow = val + j * cols;

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					sum += valRow[i] * adjRow[i];
				}

				for (int i = 0; i < cols; ++i)
				{
					if (addGrad)
					{
						gradRow[i] += valRow[i] * (adjRow[i] - sum);
					}
					else
					{
						gradRow[i] = valRow[i] * (adjRow[i] - sum);
					}
				}
			}
		}



		unsafe static public void IndexSelect(Tensor result_, Tensor src_, Tensor indice_, int rows, int cols)
		{
			float* result = (float*)CpuNativeHelpers.GetBufferStart(result_);
			float* src = (float*)CpuNativeHelpers.GetBufferStart(src_);
			float* indice = (float*)CpuNativeHelpers.GetBufferStart(indice_);

			for (int j = 0; j < rows; j++)
			{

				int srcIdx = (int)indice[j];
				float* resultRow = result + j * cols;
				float* srcRow = src + srcIdx * cols;

				for (int i = 0; i < cols; ++i)
				{
					resultRow[i] = srcRow[i];
				}
			}
		}


		unsafe static public void IndexSelectGrad(Tensor grad_, Tensor adj_, Tensor indice_, int rows, int cols)
		{
			float* grad = (float*)CpuNativeHelpers.GetBufferStart(grad_);
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* indice = (float*)CpuNativeHelpers.GetBufferStart(indice_);

			for (int j = 0; j < rows; j++)
			{

				int gradIdx = (int)indice[j];
				float* adjRow = adj + j * cols;
				float* gradRow = grad + gradIdx * cols;

				for (int i = 0; i < cols; ++i)
				{
					gradRow[i] += adjRow[i];
				}
			}
		}



		unsafe static public void LayerNorm(Tensor out_,
			Tensor in_,
			Tensor gamma_,
			Tensor beta_,
			float eps,
			int rows,
			int cols)
		{
			float* outPtr = (float*)CpuNativeHelpers.GetBufferStart(out_);
			float* inPtr = (float*)CpuNativeHelpers.GetBufferStart(in_);
			float* alpha = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
			float* beta = (beta_ != null) ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;

			for (int j = 0; j < rows; ++j)
			{
				float* so = outPtr + j * cols;
				float* sp = inPtr + j * cols;

				Span<float> spanSP = new Span<float>(sp, cols);

				float sum = 0.0f;
				//for (int i = 0; i < cols; ++i)
				//{
				//	sum += sp[i];
				//}


				int vectorSize = Vector<float>.Count;
				Vector<float> vecAdded = Vector<float>.Zero;
				int i = 0;

				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					vecAdded += vecSp;
				}
				sum = Vector.Dot(vecAdded, Vector<float>.One);
				for (; i < cols; i++)
				{
					sum += sp[i];
				}



				float mean = sum / cols;
				float sqSum = 0.0f;

				//for (int i = 0; i < cols; ++i)
				//{
				//	float ex = sp[i] - mean;
				//	sqSum += ex * ex;
				//}

				Vector<float> vecMean = new Vector<float>(mean);
				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecEx = vecSp - vecMean;
					sqSum += Vector.Dot(vecEx, vecEx);
				}
                for (; i < cols; ++i)
                {
                    float ex = sp[i] - mean;
                    sqSum += ex * ex;
                }

                float sigma = (float)Math.Sqrt(eps + sqSum / cols);

				Span<float> spanSO = new Span<float>(so, cols);
				Span<float> spanAlpha = new Span<float>(alpha, cols);
				Span<float> spanBeta = (beta != null) ? new Span<float>(beta, cols) : null;
				Vector<float> vecSigma = new Vector<float>(sigma);

				//for (int i = 0; i < cols; ++i)
				//{
				//	float t = alpha[i] * ((sp[i] - mean) / sigma);
				//	if (beta != null)
				//	{
				//		t += beta[i];
				//	}

				//	so[i] = t;
				//}


				for (i = 0; i < cols - vectorSize; i += vectorSize)
				{
					Vector<float> vecSp = new Vector<float>(spanSP.Slice(i));
					Vector<float> vecAlpha = new Vector<float>(spanAlpha.Slice(i));

					Vector<float> vecT = vecAlpha * ((vecSp - vecMean) / vecSigma);

					if (spanBeta != null)
					{
						Vector<float> vecBeta = new Vector<float>(spanBeta.Slice(i));
						vecT += vecBeta;
					}

					vecT.CopyTo(spanSO.Slice(i));
				}
                for (; i < cols; ++i)
                {
                    float t = alpha[i] * ((sp[i] - mean) / sigma);
                    if (beta != null)
                    {
                        t += beta[i];
                    }

                    so[i] = t;
                }


            }
		}


		unsafe static public void LayerNormGrad(Tensor gradX_,
			Tensor gradGamma_,
			Tensor gradBeta_,
			Tensor adj_,
			Tensor y_,
			Tensor x_,
			Tensor gamma_,
			Tensor beta_,
			int rows,
			int cols,
			float eps)
		{
			float* gradX = (float*)CpuNativeHelpers.GetBufferStart(gradX_);
			float* gradGamma = (float*)CpuNativeHelpers.GetBufferStart(gradGamma_);
			float* gradBeta = gradBeta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(gradBeta_) : null;
			float* adj = (float*)CpuNativeHelpers.GetBufferStart(adj_);
			float* y = (float*)CpuNativeHelpers.GetBufferStart(y_);
			float* x = (float*)CpuNativeHelpers.GetBufferStart(x_);
			float* gamma = (float*)CpuNativeHelpers.GetBufferStart(gamma_);
			float* beta = beta_ != null ? (float*)CpuNativeHelpers.GetBufferStart(beta_) : null;

			if (beta != null)
			{
				for (int j = 0; j < rows; ++j)
				{
					float* xRow = x + j * cols;
					float* yRow = y + j * cols;
					float* adjRow = adj + j * cols;
					float* gradXRow = gradX + j * cols;

					float sum_x = 0.0f;
					float sum_adj = 0.0f;
					float sum_adj_x = 0.0f;
					float sum_sqr = 0.0f;

					for (int i = 0; i < cols; ++i)
					{
						sum_x += xRow[i];
						sum_adj_x += adjRow[i] * (yRow[i] - (beta != null ? beta[i] : 0.0f)) / gamma[i];
						sum_adj += adjRow[i];
					}

					float mean = sum_x / cols;
					for (int i = 0; i < cols; ++i)
					{
						float ex = xRow[i] - mean;
						sum_sqr += ex * ex;
					}

					float sigma = (float)Math.Sqrt(eps + sum_sqr / cols);
					for (int i = 0; i < cols; ++i)
					{
						float grad_x = 0.0f;
						float x_hat = (yRow[i] - beta[i]) / gamma[i];
						grad_x += cols * adjRow[i];
						grad_x -= sum_adj;
						grad_x -= sum_adj_x * x_hat;
						grad_x /= cols * sigma;

						gradXRow[i] += gamma[i] * grad_x;
						gradGamma[i] += adjRow[i] * x_hat;
						gradBeta[i] += adjRow[i];
					}
				}
			}
			else
			{
				for (int j = 0; j < rows; ++j)
				{
					float* xRow = x + j * cols;
					float* yRow = y + j * cols;
					float* adjRow = adj + j * cols;
					float* gradXRow = gradX + j * cols;

					float sum_x = 0.0f;
					float sum_adj = 0.0f;
					float sum_adj_x = 0.0f;
					float sum_sqr = 0.0f;

					for (int i = 0; i < cols; ++i)
					{
						sum_x += xRow[i];
						sum_adj_x += adjRow[i] * (yRow[i] - (beta != null ? beta[i] : 0.0f)) / gamma[i];
						sum_adj += adjRow[i];
					}

					float mean = sum_x / cols;

					for (int i = 0; i < cols; ++i)
					{
						float ex = xRow[i] - mean;
						sum_sqr += ex * ex;
					}

					float sigma = (float)Math.Sqrt(eps + sum_sqr / cols);

					for (int i = 0; i < cols; ++i)
					{
						float grad_x = 0.0f;
						float x_hat = yRow[i] / gamma[i];
						grad_x += cols * adjRow[i];
						grad_x -= sum_adj;
						grad_x -= sum_adj_x * x_hat;
						grad_x /= cols * sigma;

						gradXRow[i] += gamma[i] * grad_x;
						gradGamma[i] += adjRow[i] * x_hat;
					}
				}
			}
		}


unsafe static public void Adam(Tensor tw, Tensor tg, Tensor tv, Tensor tm, int rows, int cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
		{
			float* w = (float*)CpuNativeHelpers.GetBufferStart(tw);
			float* g = (float*)CpuNativeHelpers.GetBufferStart(tg);
			float* v = (float*)CpuNativeHelpers.GetBufferStart(tv);
			float* m = (float*)CpuNativeHelpers.GetBufferStart(tm);

			for (int j = 0; j < rows; j++)
			{
				float* sw = w + j * cols;
				float* sg = g + j * cols;
				float* sv = v + j * cols;
				float* sm = m + j * cols;

				for (int i = 0; i < cols; i++)
				{
					if (sg[i] != 0.0)
					{
						float g2 = sg[i] / batchSize;

						if (g2 > clipval)
						{
							g2 = clipval;
						}
						if (g2 < -clipval)
						{
							g2 = -clipval;
						}

						sm[i] = sm[i] * decay_rate_m + (1.0f - decay_rate_m) * g2;
						sv[i] = sv[i] * decay_rate_v + (1.0f - decay_rate_v) * g2 * g2;

					    double m_cap = sm[i] / (1.0 - Math.Pow(decay_rate_m, iter));
						double v_cap = sv[i] / (1.0 - Math.Pow(decay_rate_v, iter));

						sw[i] -= (float)(step_size * m_cap / (Math.Sqrt(v_cap) + eps));

						sg[i] = 0;
					}
				}
			}
		}
	}
}