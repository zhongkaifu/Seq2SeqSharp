// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Text;

namespace TensorSharp.CUDA.DeviceCode
{
    public class PermutationGenerator
    {
        public readonly StringBuilder sb = new StringBuilder();

        public PermutationGenerator()
        {
        }

        public override string ToString()
        {
            return sb.ToString();
        }

        private string GetElementTypeString(DType dType)
        {
            if (dType == DType.Float32)
            {
                return "float";
            }
            else if (dType == DType.Float64)
            {
                return "double";
            }
            else if (dType == DType.Float16)
            {
                return "__half";
            }
            else
            {
                throw new System.NotSupportedException($"Type '{dType}' is not supported.");
            }
        }

        public void AddApplyT(string kernelBaseName, string operatorCode, DType[] elementTypes = null)
        {
            if (elementTypes == null)
            {
                elementTypes = new DType[] { DType.Float32 };
            }

            List<ApplySpecialization> specs = new List<ApplySpecialization>();
            specs.AddRange(ApplySpecialization.AllSpecializations(1, elementTypes));

            foreach (ApplySpecialization spec in specs)
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()({elementTypeA} *v) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> src, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, src);");

                if (spec.TensorElementTypes[0] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptA = (float *)src.data;");
                }
                else if (spec.TensorElementTypes[0] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptA = (__half *)src.data;");
                }

                sb.AppendLine($"         ConcreteOp_{kernelName}()(&ptA[aOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTT(string kernelBaseName, string operatorCode, DType[] elementTypes = null)
        {
            if (elementTypes == null)
            {
                elementTypes = new DType[] { DType.Float32, DType.Float32 };
            }

            List<ApplySpecialization> specs = new List<ApplySpecialization>();
            specs.AddRange(ApplySpecialization.AllSpecializations(2, elementTypes));

            foreach (ApplySpecialization spec in specs)
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");

                if (spec.TensorElementTypes[0] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptA = (float *)tensorA.data;");
                }
                else if (spec.TensorElementTypes[0] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptA = (__half *)tensorA.data;");
                }

                if (spec.TensorElementTypes[1] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptB = (float *)tensorB.data;");
                }
                else if (spec.TensorElementTypes[1] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptB = (__half *)tensorB.data;");
                }

                sb.AppendLine($"         ConcreteOp_{kernelName}()(&ptA[aOffset], &ptB[bOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");


            }
        }

        public void AddApplyTTT(string kernelBaseName, string operatorCode, DType[] elementTypes = null)
        {
            if (elementTypes == null)
            {
                elementTypes = new DType[] { DType.Float32, DType.Float32, DType.Float32 };
            }

            List<ApplySpecialization> specs = new List<ApplySpecialization>();
            specs.AddRange(ApplySpecialization.AllSpecializations(3, elementTypes));

            foreach (ApplySpecialization spec in specs)
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);
                string elementTypeC = GetElementTypeString(spec.TensorElementTypes[2]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b, {elementTypeC} *c) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");

                if (spec.TensorElementTypes[0] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptA = (float *)tensorA.data;");
                }
                else if (spec.TensorElementTypes[0] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptA = (__half *)tensorA.data;");
                }

                if (spec.TensorElementTypes[1] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptB = (float *)tensorB.data;");
                }
                else if (spec.TensorElementTypes[1] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptB = (__half *)tensorB.data;");
                }

                if (spec.TensorElementTypes[2] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptC = (float *)tensorC.data;");
                }
                else if (spec.TensorElementTypes[2] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptC = (__half *)tensorC.data;");
                }

                sb.AppendLine($"         ConcreteOp_{kernelName}()(&ptA[aOffset], &ptB[bOffset], &ptC[cOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");

            }
        }

        public void AddApplyTTTT(string kernelBaseName, string operatorCode, DType[] elementTypes = null)
        {
            if (elementTypes == null)
            {
                elementTypes = new DType[] { DType.Float32, DType.Float32, DType.Float32, DType.Float32 };
            }

            List<ApplySpecialization> specs = new List<ApplySpecialization>();
            specs.AddRange(ApplySpecialization.AllSpecializations(4, elementTypes));

            foreach (ApplySpecialization spec in specs)
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();
                string dimsD = spec.TensorDims[3].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);
                string elementTypeC = GetElementTypeString(spec.TensorElementTypes[2]);
                string elementTypeD = GetElementTypeString(spec.TensorElementTypes[3]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b, {elementTypeC} *c, {elementTypeD} *d) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);");

                if (spec.TensorElementTypes[0] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptA = (float *)tensorA.data;");
                }
                else if (spec.TensorElementTypes[0] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptA = (__half *)tensorA.data;");
                }

                if (spec.TensorElementTypes[1] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptB = (float *)tensorB.data;");
                }
                else if (spec.TensorElementTypes[1] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptB = (__half *)tensorB.data;");
                }

                if (spec.TensorElementTypes[2] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptC = (float *)tensorC.data;");
                }
                else if (spec.TensorElementTypes[2] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptC = (__half *)tensorC.data;");
                }

                if (spec.TensorElementTypes[3] == DType.Float32)
                {
                    sb.AppendLine($"     float *ptD = (float *)tensorD.data;");
                }
                else if (spec.TensorElementTypes[3] == DType.Float16)
                {
                    sb.AppendLine($"     __half *ptD = (__half *)tensorD.data;");
                }

                sb.AppendLine($"         ConcreteOp_{kernelName}()(&ptA[aOffset], &ptB[bOffset], &ptC[cOffset], &ptD[dOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(5))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();
                string dimsD = spec.TensorDims[3].ToString();
                string dimsE = spec.TensorDims[4].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);
                string elementTypeC = GetElementTypeString(spec.TensorElementTypes[2]);
                string elementTypeD = GetElementTypeString(spec.TensorElementTypes[3]);
                string elementTypeE = GetElementTypeString(spec.TensorElementTypes[4]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b, {elementTypeC} *c, {elementTypeD} *d, {elementTypeE} *e) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, TensorInfo<{indexType}> tensorE, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);");
                sb.AppendLine($"         const {indexType} eOffset = IndexToOffset < {indexType}, {dimsE}>::get(linearIndex, tensorE);");
                sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset], &tensorE.data[eOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }
        public void AddApplyTS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1, new DType[] { DType.Float32 }))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float b;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal) {{ this->b = bVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a) const {{ {operatorCode} }}");
                sb.AppendLine("};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset<{indexType}, {dimsA}>::get(linearIndex, a);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b);");
                sb.AppendLine($"         op(&a.data[aOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTSHalf(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1, new DType[] { DType.Float16 }))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("__half b;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal) {{ this->b = __float2half(bVal); }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a) const {{ {operatorCode} }}");
                sb.AppendLine("};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset<{indexType}, {dimsA}>::get(linearIndex, a);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b);");
                sb.AppendLine($"         __half *pt = (__half *)(a.data);");
                sb.AppendLine($"         op(&pt[aOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTSS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float b;");
                sb.AppendLine("float c;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal, float cVal) {{ this->b = bVal; this->c = cVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a) const {{ {operatorCode} }}");
                sb.AppendLine("};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, float c, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b, c);");
                sb.AppendLine($"         op(&a.data[aOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTTS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);


                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float c;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal) {{ this->c = cVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b) const {{ {operatorCode} }} }};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c);");
                sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");

            }
        }

        public void AddApplyTTSHalf(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2, new DType[] {DType.Float16, DType.Float16 }))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);


                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("__half c;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal) {{ this->c = __float2half(cVal); }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b) const {{ {operatorCode} }} }};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c);");
                sb.AppendLine($"         __half *ptA = (__half *)(tensorA.data);");
                sb.AppendLine($"         __half *ptB = (__half *)(tensorB.data);");
                sb.AppendLine($"         op(&ptA[aOffset], &ptB[bOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");

            }
        }

        public void AddApplyTTSS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float c;");
                sb.AppendLine("float d;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal, float dVal) {{ this->c = cVal; this->d = dVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b) const {{ {operatorCode} }} }};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, float c, float d, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(c, d);");
                sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTTTS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(3))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);
                string elementTypeC = GetElementTypeString(spec.TensorElementTypes[2]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float d;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float dVal) {{ this->d = dVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b, {elementTypeC} *c) const {{ {operatorCode} }} }};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, float d, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(d);");
                sb.AppendLine($"         op(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTTTSHalf(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(3, new DType[] { DType.Float16, DType.Float16, DType.Float16 }))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();

                string elementTypeA = GetElementTypeString(spec.TensorElementTypes[0]);
                string elementTypeB = GetElementTypeString(spec.TensorElementTypes[1]);
                string elementTypeC = GetElementTypeString(spec.TensorElementTypes[2]);

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("__half d;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float dVal) {{ this->d = __float2half(dVal); }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()({elementTypeA} *a, {elementTypeB} *b, {elementTypeC} *c) const {{ {operatorCode} }} }};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, float d, __int64 totalElements)");
                sb.AppendLine("   {");
                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(d);");
                sb.AppendLine($"         __half *ptA = (__half *)(tensorA.data);");
                sb.AppendLine($"         __half *ptB = (__half *)(tensorB.data);");
                sb.AppendLine($"         __half *ptC = (__half *)(tensorC.data);");
                sb.AppendLine($"         op(&ptA[aOffset], &ptB[bOffset], &ptC[cOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddReduce(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_KERNELS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceNorm(string kernelBaseName)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_NORM_KERNELS({0}, {1}, {2}, {3})\n", indexType, dimsA, dimsB, kernelName);
            }
        }

        public void AddReduceAll(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_KERNELS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceAllNorm(string kernelBaseName)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_NORM_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }

        public void AddReduceAllSubSquare(string kernelBaseName)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_SUB_SQUARE_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }


        // TODO make member of ApplySpecialization
        public static string GetMangledName(string baseName, ApplySpecialization spec)
        {
            StringBuilder sb = new StringBuilder();

            sb.Append(baseName);
            sb.Append(spec.Use32BitIndices ? "__int32" : "__int64");
            foreach (int dimSize in spec.TensorDims)
            {
                sb.Append("_").Append(dimSize.ToString().Replace('-', 'M'));
            }

            foreach (var dtype in spec.TensorElementTypes)
            {
                sb.Append("_").Append(dtype.ToString().Replace('-', 'M'));
            }

            return sb.ToString();
        }
    }
}
