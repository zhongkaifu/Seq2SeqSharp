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

        public void AddApplyT(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* v) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> src, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, src);");
                sb.AppendLine($"         ConcreteOp_{kernelName}()(&src.data[aOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");
            }
        }

        public void AddApplyTT(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();


                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");


            }
        }

        public void AddApplyTTT(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(3))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset]);");
                sb.AppendLine("      }");
                sb.AppendLine("   }");
                sb.AppendLine("}");

            }
        }

        public void AddApplyTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(4))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();
                string dimsC = spec.TensorDims[2].ToString();
                string dimsD = spec.TensorDims[3].ToString();

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d) const {{ {operatorCode} }} }};");
                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> tensorA, TensorInfo<{indexType}> tensorB, TensorInfo<{indexType}> tensorC, TensorInfo<{indexType}> tensorD, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, tensorA);");
                sb.AppendLine($"         const {indexType} bOffset = IndexToOffset < {indexType}, {dimsB}>::get(linearIndex, tensorB);");
                sb.AppendLine($"         const {indexType} cOffset = IndexToOffset < {indexType}, {dimsC}>::get(linearIndex, tensorC);");
                sb.AppendLine($"         const {indexType} dOffset = IndexToOffset < {indexType}, {dimsD}>::get(linearIndex, tensorD);");
                sb.AppendLine($"         ConcreteOp_{kernelName}()(&tensorA.data[aOffset], &tensorB.data[bOffset], &tensorC.data[cOffset], &tensorD.data[dOffset]);");
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

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{ __device__ __forceinline__ void operator()(float* a, float *b, float *c, float *d, float *e) const {{ {operatorCode} }} }};");
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
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(1))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float b;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal) {{ this->b = bVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}");
                sb.AppendLine("};");

                sb.AppendLine("extern \"C\" {");
                sb.AppendLine($"   __global__ void {kernelName}(TensorInfo<{indexType}> a, float b, __int64 totalElements)");
                sb.AppendLine("   {");

                sb.AppendLine($"      for ({indexType} linearIndex = blockIdx.x * blockDim.x + threadIdx.x;linearIndex < totalElements;linearIndex += gridDim.x * blockDim.x)");
                sb.AppendLine("      {");
                sb.AppendLine($"         const {indexType} aOffset = IndexToOffset < {indexType}, {dimsA}>::get(linearIndex, a);");
                sb.AppendLine($"         ConcreteOp_{kernelName} op = ConcreteOp_{kernelName}(b);");
                sb.AppendLine($"         op(&a.data[aOffset]);");
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

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float b;");
                sb.AppendLine("float c;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float bVal, float cVal) {{ this->b = bVal; this->c = cVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()(float* a) const {{ {operatorCode} }}");
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

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float c;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal) {{ this->c = cVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");

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

        public void AddApplyTTSS(string kernelBaseName, string operatorCode)
        {
            foreach (ApplySpecialization spec in ApplySpecialization.AllSpecializations(2))
            {
                string kernelName = GetMangledName(kernelBaseName, spec);
                string indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                string dimsA = spec.TensorDims[0].ToString();
                string dimsB = spec.TensorDims[1].ToString();

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float c;");
                sb.AppendLine("float d;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float cVal, float dVal) {{ this->c = cVal; this->d = dVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b) const {{ {operatorCode} }} }};");

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

                sb.AppendLine($"struct ConcreteOp_{kernelName} {{");
                sb.AppendLine("float d;");
                sb.AppendLine($"__device__ ConcreteOp_{kernelName}(float dVal) {{ this->d = dVal; }}");
                sb.AppendLine($"__device__ __forceinline__ void operator()(float* a, float *b, float *c) const {{ {operatorCode} }} }};");

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
            return sb.ToString();
        }
    }
}
