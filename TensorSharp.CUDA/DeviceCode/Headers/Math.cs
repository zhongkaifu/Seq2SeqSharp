// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "Math")]
    public static class MathHeader
    {
		public static string Code = (TSCudaContext.ElementType == DType.Float16) ? Code32 + Code16 : Code32;


        public const string Code32 = @"
#define INLINE_FUNC __device__ __forceinline__

//INLINE_FUNC uint8 Mod_op(uint8 x, uint8 y) { return x % y; }
INLINE_FUNC __int32 Mod_op(__int32 x, __int32 y) { return x % y; }
INLINE_FUNC float Mod_op(float x, float y) { return fmod(x, y); }
INLINE_FUNC double Mod_op(double x, double y) { return fmod(x, y); }

template<typename T> INLINE_FUNC T rsub_op(T x, T y) { return (T)(y - x); }
template<typename T> INLINE_FUNC T rdiv_op(T x, T y) { return (T)(y / x); }

#define INFIX_TO_FUNC(OPNAME, OPERATOR) template<typename T> INLINE_FUNC T OPNAME(T x, T y) { return (T)(x OPERATOR y); }
INFIX_TO_FUNC(add_op, +)
INFIX_TO_FUNC(sub_op, -)
INFIX_TO_FUNC(mul_op, *)
INFIX_TO_FUNC(div_op, /)

INFIX_TO_FUNC(gt_op, >)
INFIX_TO_FUNC(lt_op, <)
INFIX_TO_FUNC(ge_op, >=)
INFIX_TO_FUNC(le_op, <=)
INFIX_TO_FUNC(eq_op, ==)
INFIX_TO_FUNC(ne_op, !=)


template<typename T> INLINE_FUNC T Neg(T x) {
	return -x;
}

template<typename T> INLINE_FUNC T AddDiv(T x, T y, T z) {
	return x + y / z;
}


template<typename T> INLINE_FUNC T AddMul(T x, T y, T z) {
	return x + y * z;
}

template<typename T> INLINE_FUNC T MulMulAdd(T x, T y, T z, T w) {
	return x * y + z * w;
}

template<typename T> INLINE_FUNC T Frac(T x) {
	return x - trunc(x);
}

template<typename T> INLINE_FUNC T Lerp(T a, T b, T weight) {
	return a + weight * (b - a);
}


template<typename T> INLINE_FUNC T SiLU(T w) {
	return w / (T(1) + expf(-w));
}

template<typename T> INLINE_FUNC T SiLUD(T w, T resG) {

  T sig = T(1) / (T(1) + expf(-w));
  T grad = sig * (T(1) + w * (T(1) - sig));
  return resG * grad;
}

template<typename T> INLINE_FUNC T AddSiLUD(T t, T w, T resG) {

  T sig = T(1) / (T(1) + expf(-w));
  T grad = sig * (T(1) + w * (T(1) - sig));
  return t + resG * grad;

}


template<typename T> INLINE_FUNC T Sigmoid(T x) {
	return T(1) / (T(1) + expf(-x));
}

template<typename T> INLINE_FUNC T AddSigmoidD(T t, T resW, T resG) {
	return t + resW * (T(1) - resW) * resG;
}


template<typename T> INLINE_FUNC T AddTanhD(T t, T resW, T resG) {
	return t + (T(1) - resW * resW) * resG;
}


template<typename T> INLINE_FUNC T SigmoidD(T resW, T resG) {
	return resW * (T(1) - resW) * resG;
}


template<typename T> INLINE_FUNC T TanhD(T resW, T resG) {
	return (T(1) - resW * resW) * resG;
}


template<typename T> INLINE_FUNC T AddTanh(T x, T y) {
	return tanhf(x + y);
}


template<typename T> INLINE_FUNC T AddTanh3(T x, T y, T z) {
	return tanhf(x + y + z);
}


template <typename T> INLINE_FUNC T sgn(T val) {
	if (val < T(0))
		return T(-1);
	if (val > T(0))
		return T(1);
	return T(0);
}

template <typename T> INLINE_FUNC T relu(T w) {
	if (w < T(0))
		return T(0);
	return w;
}


template <typename T> INLINE_FUNC T relud(T w, T g) {
	if (w > T(0))
		return g;
	return T(0);
}


template <typename T> INLINE_FUNC T addrelud(T t, T w, T g) {
	if (w > T(0))
		return t + g;
	return t;
}



template <typename T> INLINE_FUNC T LeakyReLU(T w) {
	if (w < T(0))
		return T(0.01) * w;
	return w;
}


template <typename T> INLINE_FUNC T LeakyReLUD(T w, T g) {
	if (w >= T(0))
		return g;
	return T(0.01) * g;
}


template <typename T> INLINE_FUNC T AddLeakyReLUD(T t, T w, T g) {
	if (w >= T(0))
		return t + g;
	return t + T(0.01) * g;
}


template <typename T> INLINE_FUNC T Clamp(T val, T min, T max) {
	if (val < min)
		return min;
	if (val > max)
		return max;
	return val;
}

template <typename T> INLINE_FUNC T MaskFill(T t, T mask, T defValue) {
	if (mask == T(0))
		return t;
	return defValue;
}


";

        public const string Code16 = @"
#include <cuda_fp16.h>

template<typename T> INLINE_FUNC T SiLUHalf(T wh) {
    float w = __half2float(wh);
	float res = w / (1.0 + expf(-w));
    return __float2half(res);
}

template<typename T> INLINE_FUNC T SiLUDHalf(T wh, T resGh) {

  float w = __half2float(wh);
  float resG = __half2float(resGh);

  float sig = 1.0 / (1.0 + expf(-w));
  float grad = sig * (1.0 + w * (1.0 - sig));
  return __float2half(resG * grad);
}

template<typename T> INLINE_FUNC T AddSiLUDHalf(T th, T wh, T resGh) {

  float t = __half2float(th);
  float w = __half2float(wh);
  float resG = __half2float(resGh);

  float sig = 1.0 / (1.0 + expf(-w));
  float grad = sig * (1.0 + w * (1.0 - sig));
  return __float2half(t + resG * grad);

}

template <typename T> INLINE_FUNC T addreludhalf(T t, T w, T g) {
	if (w > T(0))
		return __hadd(t, g);
	return t;
}

template <typename T> INLINE_FUNC T LeakyReLUHalf(T w) {
	if (w < T(0))
		return __hmul(T(0.01), w);
	return w;
}


template <typename T> INLINE_FUNC T LeakyReLUDHalf(T w, T g) {
	if (w >= T(0))
		return g;
	return __hmul(T(0.01), g);
}


template <typename T> INLINE_FUNC T AddLeakyReLUDHalf(T t, T w, T g) {
	if (w >= T(0))
		return __hadd(t, g);

	return __hadd(t, __hmul(T(0.01), g));
}



";
    }
}
