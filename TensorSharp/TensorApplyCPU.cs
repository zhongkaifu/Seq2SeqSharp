using System;
using System.Collections.Generic;
using System.Linq;
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


		unsafe public static void CAdd_Apply(Tensor result, Tensor lhs, Tensor rhs)
		{
			unsafe void func(float* r, float* left, float* right)
			{
				*r = add(*left, *right);
			}

			Apply3(result, lhs, rhs, func);
		}

		unsafe public static void TSMul_Apply(Tensor result, Tensor src, float value)
		{
			unsafe void func(float* r, float* s)
			{
				*r = mul(*s, value);
			}

			Apply2(result, src, func);
		}

		static float relu(float w)
		{
			if (w < 0.0f)
				return 0.0f;
			return w;

		}

		static float add(float x, float y)
		{
			return x + y;
		}

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

				for (int i = 0; i < cols; ++i)
				{
					so[i] /= sum;
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

				float sum = 0.0f;
				for (int i = 0; i < cols; ++i)
				{
					sum += sp[i];
				}

				float mean = sum / cols;
				float sqSum = 0.0f;

				for (int i = 0; i < cols; ++i)
				{
					float ex = sp[i] - mean;
					sqSum += ex * ex;
				}

				float sigma = (float)Math.Sqrt(eps + sqSum / cols);

				for (int i = 0; i < cols; ++i)
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
	}
}
