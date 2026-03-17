using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView2D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly long RawBytes;

        public GgmlTensorView2D(IntPtr data, int dim0, int dim1, int stride0, int stride1, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Stride0 = stride0;
            Stride1 = stride1;
            RawBytes = rawBytes;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView3D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Dim2;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly int Stride2;
        public readonly long RawBytes;

        public GgmlTensorView3D(IntPtr data, int dim0, int dim1, int dim2, int stride0, int stride1, int stride2, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Dim2 = dim2;
            Stride0 = stride0;
            Stride1 = stride1;
            Stride2 = stride2;
            RawBytes = rawBytes;
        }
    }

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlTensorView4D
{
    public readonly IntPtr Data;
    public readonly int Ne0;
    public readonly int Ne1;
    public readonly int Ne2;
    public readonly int Ne3;
    public readonly long Nb1;
    public readonly long Nb2;
    public readonly long Nb3;
    public readonly long RawBytes;

    public GgmlTensorView4D(IntPtr data, int ne0, int ne1, int ne2, int ne3, long nb1, long nb2, long nb3, long rawBytes)
    {
        Data = data;
        Ne0 = ne0;
        Ne1 = ne1;
        Ne2 = ne2;
        Ne3 = ne3;
        Nb1 = nb1;
        Nb2 = nb2;
        Nb3 = nb3;
        RawBytes = rawBytes;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlContiguousTensor
{
    public readonly IntPtr Data;
    public readonly long ElementCount;

    public GgmlContiguousTensor(IntPtr data, long elementCount)
    {
        Data = data;
        ElementCount = elementCount;
    }
}

internal enum GgmlUnaryOp
{
    Neg = 1,
    Exp = 2,
    Log = 3,
    Sqrt = 4,
    Relu = 5,
    Sigmoid = 6,
    Tanh = 7,
    SiLU = 8,
    Step = 9,
    Abs = 10,
    Sign = 11,
}

internal enum GgmlBinaryTensorOp
{
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
}

internal enum GgmlBinaryScalarOp
{
    Add = 1,
    Sub = 2,
    ReverseSub = 3,
    Mul = 4,
    Div = 5,
    ReverseDiv = 6,
}

internal enum GgmlActivationGradOp
{
    Relu = 1,
    Sigmoid = 2,
    Tanh = 3,
    SiLU = 4,
}

internal enum GgmlNormOp
{
    LayerNorm = 1,
    RmsNorm = 2,
}

internal enum GgmlReductionOp
{
    Sum = 1,
    Mean = 2,
}

internal enum GgmlIndexReductionOp
{
    Argmin = 1,
    Argmax = 2,
}

    internal static class GgmlNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention CallingConventionType = CallingConvention.Cdecl;

        static GgmlNative()
        {
            NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_GetLastError();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsMetalAvailable();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlTensorView2D m1,
            GgmlTensorView2D m2,
            float beta,
            float alpha);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmBatchF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView3D m1,
            GgmlTensorView3D m2,
            float beta,
            float alpha);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ReduceLastDimF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexReductionF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxGradF32(
            GgmlTensorView4D result,
            GgmlTensorView4D adj,
            GgmlTensorView4D val,
            int addGrad);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossF32(
            out float lossValue,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float smooth,
            float labelSmooth);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossBackwardF32(
            GgmlTensorView4D grad,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float lossGradient,
            float smooth,
            float labelSmooth,
            int addGrad);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AdamF32(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CopyF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_UnaryF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryTensorF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D lhs,
            GgmlTensorView4D rhs);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryScalarF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            float scalar);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ActivationGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D grad,
            GgmlTensorView4D accumulation,
            int hasAccumulation);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D gamma,
            GgmlTensorView4D beta,
            int hasBeta,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D gradGamma,
            GgmlTensorView4D gradBeta,
            GgmlTensorView4D adj,
            GgmlTensorView4D x,
            GgmlTensorView4D gamma,
            int hasGradBeta,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlContiguousTensor indices,
            int addToResult);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectGradF32(
            GgmlTensorView2D grad,
            GgmlTensorView2D adj,
            GgmlContiguousTensor indices);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            int seqLen,
            int rowOffset,
            int addToResult,
            int invertPositions);

        public static void EnsureAvailable()
        {
            if (!OperatingSystem.IsMacOS())
            {
                throw new PlatformNotSupportedException("The GGML Metal backend is available on macOS only.");
            }

            try
            {
                if (TSGgml_IsMetalAvailable() == 0)
                {
                    throw new InvalidOperationException($"Failed to initialize ggml-metal. {GetLastErrorMessage("Build the native GGML bridge and ensure Metal is available on this Mac.")}");
                }
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException("Failed to load the native GGML bridge. Build `TensorSharp.GGML.Native` first.", ex);
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException("The native GGML bridge is out of date. Rebuild `TensorSharp.GGML.Native`.", ex);
            }
        }

        public static void Addmm(GgmlTensorView2D result, GgmlTensorView2D src, GgmlTensorView2D m1, GgmlTensorView2D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmF32(result, src, m1, m2, beta, alpha), "addmm");
        }

        public static void AddmmBatch(GgmlTensorView3D result, GgmlTensorView3D src, GgmlTensorView3D m1, GgmlTensorView3D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmBatchF32(result, src, m1, m2, beta, alpha), "addmmbatch");
        }

        public static void ReduceLastDim(GgmlReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_ReduceLastDimF32((int)op, result, src), op.ToString());
        }

        public static void IndexReduction(GgmlIndexReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_IndexReductionF32((int)op, result, src), op.ToString());
        }

        public static void Softmax(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_SoftmaxF32(result, src), "softmax");
        }

        public static void SoftmaxGrad(GgmlTensorView4D result, GgmlTensorView4D adj, GgmlTensorView4D val, bool addGrad)
        {
            CheckResult(TSGgml_SoftmaxGradF32(result, adj, val, addGrad ? 1 : 0), "softmaxgrad");
        }

        public static float CrossEntropyLoss(GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float smooth, float labelSmooth)
        {
            CheckResult(TSGgml_CrossEntropyLossF32(out float lossValue, probs, targetIndices, smooth, labelSmooth), "crossentropyloss");
            return lossValue;
        }

        public static void CrossEntropyLossBackward(GgmlTensorView4D grad, GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float lossGradient, float smooth, float labelSmooth, bool addGrad)
        {
            CheckResult(TSGgml_CrossEntropyLossBackwardF32(grad, probs, targetIndices, lossGradient, smooth, labelSmooth, addGrad ? 1 : 0), "crossentropyloss_backward");
        }

        public static void Adam(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            CheckResult(TSGgml_AdamF32(weight, gradient, v, m, gradNormFactor, stepSize, clipValue, regc, decayRateV, decayRateM, iter, eps), "adam");
        }

        public static void Copy(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_CopyF32(result, src), "copy");
        }

        public static void Unary(GgmlUnaryOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_UnaryF32((int)op, result, src), op.ToString());
        }

        public static void BinaryTensor(GgmlBinaryTensorOp op, GgmlTensorView4D result, GgmlTensorView4D lhs, GgmlTensorView4D rhs)
        {
            CheckResult(TSGgml_BinaryTensorF32((int)op, result, lhs, rhs), op.ToString());
        }

        public static void BinaryScalar(GgmlBinaryScalarOp op, GgmlTensorView4D result, GgmlTensorView4D src, float scalar)
        {
            CheckResult(TSGgml_BinaryScalarF32((int)op, result, src, scalar), op.ToString());
        }

        public static void ActivationGrad(GgmlActivationGradOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D grad, GgmlTensorView4D accumulation, bool hasAccumulation)
        {
            CheckResult(TSGgml_ActivationGradF32((int)op, result, src, grad, accumulation, hasAccumulation ? 1 : 0), $"{op}Grad");
        }

        public static void Norm(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D gamma, GgmlTensorView4D beta, bool hasBeta, float eps)
        {
            CheckResult(TSGgml_NormF32((int)op, result, src, gamma, beta, hasBeta ? 1 : 0, eps), op.ToString());
        }

        public static void NormGrad(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D gradGamma, GgmlTensorView4D gradBeta, GgmlTensorView4D adj, GgmlTensorView4D x, GgmlTensorView4D gamma, bool hasGradBeta, float eps)
        {
            CheckResult(TSGgml_NormGradF32((int)op, result, gradGamma, gradBeta, adj, x, gamma, hasGradBeta ? 1 : 0, eps), $"{op}Grad");
        }

        public static void IndexSelect(GgmlTensorView2D result, GgmlTensorView2D src, GgmlContiguousTensor indices, bool addToResult)
        {
            CheckResult(TSGgml_IndexSelectF32(result, src, indices, addToResult ? 1 : 0), "indexselect");
        }

        public static void IndexSelectGrad(GgmlTensorView2D grad, GgmlTensorView2D adj, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_IndexSelectGradF32(grad, adj, indices), "indexselectgrad");
        }

        public static void RoPE(GgmlTensorView4D result, GgmlTensorView4D src, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, src, seqLen, rowOffset, 0, 0), "rope");
        }

        public static void RoPEGrad(GgmlTensorView4D result, GgmlTensorView4D adj, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, adj, seqLen, rowOffset, 1, 1), "ropegrad");
        }

        private static void CheckResult(int result, string opName)
        {
            if (result != 0)
            {
                return;
            }

            throw new InvalidOperationException($"Native GGML {opName} failed. {GetLastErrorMessage("Unknown native GGML error.")}");
        }

        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, DllName, StringComparison.Ordinal))
            {
                return IntPtr.Zero;
            }

            foreach (string candidate in GetCandidatePaths(assembly))
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                {
                    return handle;
                }
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> GetCandidatePaths(Assembly assembly)
        {
            string baseDirectory = AppContext.BaseDirectory;
            yield return Path.Combine(baseDirectory, "libGgmlOps.dylib");
            yield return Path.Combine(Path.GetDirectoryName(assembly.Location) ?? baseDirectory, "libGgmlOps.dylib");

            foreach (string root in EnumerateRepoRoots(baseDirectory))
            {
                yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", "libGgmlOps.dylib");
                yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", "Release", "libGgmlOps.dylib");
            }
        }

        private static IEnumerable<string> EnumerateRepoRoots(string startDirectory)
        {
            DirectoryInfo current = new DirectoryInfo(startDirectory);
            while (current != null)
            {
                if (File.Exists(Path.Combine(current.FullName, "Seq2SeqSharp.sln")))
                {
                    yield return current.FullName;
                }

                current = current.Parent;
            }
        }

        private static string GetLastErrorMessage(string fallback)
        {
            IntPtr errPtr = TSGgml_GetLastError();
            string message = errPtr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(errPtr);
            return string.IsNullOrWhiteSpace(message) ? fallback : message;
        }
    }
}
