// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Tests
{
    /// <summary>
    /// Compares GGML Metal backend op results with CPU backend. Each test runs on both contiguous
    /// and non-contiguous tensors. Requires macOS for GGML Metal.
    /// </summary>
    [TestClass]
    public class GgmlOpsComparison_Tests
    {
        private const float Tolerance = 1e-4f;
        private static readonly Random Rnd = new Random(42);

        private static bool IsMacOS => RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

        private static Tensor CreateTensor(ProcessorTypeEnums backend, long[] shape, float[] values, int deviceId = 0)
        {
            TensorAllocator.InitDevices(backend, new[] { deviceId });
            var t = new Tensor(TensorAllocator.Allocator(deviceId), DType.Float32, shape);
            if (values != null)
                t.SetElementsAsFloat(values);
            return t;
        }

        private static float[] ReadTensor(Tensor t)
        {
            var arr = new float[t.ElementCount()];
            t.CopyToArray(arr);
            return arr;
        }

        private static void AssertTensorsClose(float[] cpu, float[] ggml, string opName, string layout)
        {
            Assert.AreEqual(cpu.Length, ggml.Length, $"{opName} ({layout}): length mismatch");
            for (int i = 0; i < cpu.Length; i++)
            {
                if (float.IsNaN(cpu[i]) && float.IsNaN(ggml[i])) continue;
                if (float.IsInfinity(cpu[i]) && float.IsInfinity(ggml[i]) && Math.Sign(cpu[i]) == Math.Sign(ggml[i])) continue;
                Assert.IsTrue(Math.Abs(cpu[i] - ggml[i]) <= Tolerance,
                    $"{opName} ({layout}) at [{i}]: CPU={cpu[i]:G6} GGML={ggml[i]:G6}");
            }
        }

        private static void CompareBackends(string opName, bool contiguous,
            Func<ProcessorTypeEnums, Tensor> runCpu,
            Func<ProcessorTypeEnums, Tensor> runGgml)
        {
            TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new[] { 0 });
            using var cpuResult = runCpu(ProcessorTypeEnums.CPU);
            var cpuArr = ReadTensor(cpuResult);

            TensorAllocator.InitDevices(ProcessorTypeEnums.GGML_METAL, new[] { 0 });
            using var ggmlResult = runGgml(ProcessorTypeEnums.GGML_METAL);
            var ggmlArr = ReadTensor(ggmlResult);

            AssertTensorsClose(cpuArr, ggmlArr, opName, contiguous ? "contiguous" : "non-contiguous");
        }

        [TestMethod]
        public void Ggml_Copy_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Copy", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Copy(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Copy(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Copy_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Copy", false,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var transposed = src.Transpose(); var r = CreateTensor(b, new long[] { 3, 2 }, null); Ops.Copy(r, transposed); src.Dispose(); transposed.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var transposed = src.Transpose(); var r = CreateTensor(b, new long[] { 3, 2 }, null); Ops.Copy(r, transposed); src.Dispose(); transposed.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Fill_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            CompareBackends("Fill", true,
                b => { var r = CreateTensor(b, new long[] { 3, 4 }, null); Ops.Fill(r, 2.5f); return r; },
                b => { var r = CreateTensor(b, new long[] { 3, 4 }, null); Ops.Fill(r, 2.5f); return r; });
        }

        [TestMethod]
        public void Ggml_Fill_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            CompareBackends("Fill", false,
                b => { var baseT = CreateTensor(b, new long[] { 4, 3 }, null); var view = baseT.Transpose(); Ops.Fill(view, 2.5f); return Ops.AsContiguous(view); },
                b => { var baseT = CreateTensor(b, new long[] { 4, 3 }, null); var view = baseT.Transpose(); Ops.Fill(view, 2.5f); return Ops.AsContiguous(view); });
        }

        [TestMethod]
        public void Ggml_Unary_Abs_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { -1, 2, -3, 4, -5, 6 };
            CompareBackends("Abs", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Abs(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Abs(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Abs_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { -1, 2, -3, 4, -5, 6 };
            CompareBackends("Abs", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Abs(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Abs(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Neg_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, -2, 3, -4, 5, -6 };
            CompareBackends("Neg", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Neg(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Neg(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Neg_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, -2, 3, -4, 5, -6 };
            CompareBackends("Neg", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Neg(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Neg(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Sqrt_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 4, 9, 16, 25, 36 };
            CompareBackends("Sqrt", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sqrt(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sqrt(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Sqrt_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 4, 9, 16, 25, 36 };
            CompareBackends("Sqrt", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sqrt(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sqrt(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Exp_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 2, 0.5f, -0.5f };
            CompareBackends("Exp", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Exp(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Exp(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Exp_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 2, 0.5f, -0.5f };
            CompareBackends("Exp", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Exp(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Exp(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Log_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Log", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Log(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Log(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Log_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Log", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Log(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Log(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Relu_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { -1, 2, -3, 4, 0, 6 };
            CompareBackends("Relu", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Relu(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Relu(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Relu_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { -1, 2, -3, 4, 0, 6 };
            CompareBackends("Relu", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Relu(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Relu(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Sigmoid_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 2, 0.5f, -2 };
            CompareBackends("Sigmoid", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sigmoid(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sigmoid(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Sigmoid_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 2, 0.5f, -2 };
            CompareBackends("Sigmoid", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sigmoid(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Sigmoid(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Tanh_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 0.5f, -0.5f, 2 };
            CompareBackends("Tanh", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Tanh(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Tanh(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_Tanh_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 0.5f, -0.5f, 2 };
            CompareBackends("Tanh", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Tanh(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.Tanh(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_SiLU_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 0.5f, -0.5f, 2 };
            CompareBackends("SiLU", true,
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.SiLU(r, src); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 2, 3 }, vals); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.SiLU(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Unary_SiLU_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 0, 1, -1, 0.5f, -0.5f, 2 };
            CompareBackends("SiLU", false,
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.SiLU(r, view); src.Dispose(); return r; },
                b => { var src = CreateTensor(b, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(b, new long[] { 2, 3 }, null); Ops.SiLU(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Add_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            var b = new float[] { 10, 20, 30, 40, 50, 60 };
            CompareBackends("Add", true,
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Add_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            var b = new float[] { 10, 20, 30, 40, 50, 60 };
            CompareBackends("Add", false,
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Sub_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 10, 20, 30, 40, 50, 60 };
            var b = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Sub", true,
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Sub(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Sub(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Sub_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 10, 20, 30, 40, 50, 60 };
            var b = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Sub", false,
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Sub(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Sub(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Mul_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            var b = new float[] { 2, 3, 4, 5, 6, 7 };
            CompareBackends("Mul", true,
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Mul_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            var b = new float[] { 2, 3, 4, 5, 6, 7 };
            CompareBackends("Mul", false,
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Div_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 10, 20, 30, 40, 50, 60 };
            var b = new float[] { 2, 4, 5, 5, 10, 6 };
            CompareBackends("Div", true,
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Div(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 2, 3 }, a); var rhs = CreateTensor(be, new long[] { 2, 3 }, b); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Div(r, lhs, rhs); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Div_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 10, 20, 30, 40, 50, 60 };
            var b = new float[] { 2, 4, 5, 5, 10, 6 };
            CompareBackends("Div", false,
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Div(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; },
                be => { var lhs = CreateTensor(be, new long[] { 3, 2 }, a); var rhs = CreateTensor(be, new long[] { 3, 2 }, b); var lv = lhs.Transpose(); var rv = rhs.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Div(r, lv, rv); lhs.Dispose(); rhs.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_AddScalar_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("AddScalar", true,
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, a); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, src, 10f); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, a); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, src, 10f); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_AddScalar_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("AddScalar", false,
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, a); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, view, 10f); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, a); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Add(r, view, 10f); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_MulScalar_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("MulScalar", true,
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, a); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, src, 3f); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, a); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, src, 3f); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_MulScalar_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var a = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("MulScalar", false,
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, a); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, view, 3f); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, a); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Mul(r, view, 3f); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Softmax_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            CompareBackends("Softmax", true,
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.Softmax(r, src); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.Softmax(r, src); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Softmax_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            CompareBackends("Softmax", false,
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.Softmax(r, view); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.Softmax(r, view); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Addmm_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var m1 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // 2x4
            var m2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // 4x3
            CompareBackends("Addmm", true,
                be => { var a = CreateTensor(be, new long[] { 2, 4 }, m1); var b = CreateTensor(be, new long[] { 4, 3 }, m2); var src = CreateTensor(be, new long[] { 2, 3 }, new float[6]); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Addmm(r, 0f, src, 1f, a, b); a.Dispose(); b.Dispose(); src.Dispose(); return r; },
                be => { var a = CreateTensor(be, new long[] { 2, 4 }, m1); var b = CreateTensor(be, new long[] { 4, 3 }, m2); var src = CreateTensor(be, new long[] { 2, 3 }, new float[6]); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Addmm(r, 0f, src, 1f, a, b); a.Dispose(); b.Dispose(); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Addmm_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var m1Base = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // 2x4 -> transpose to 4x2, but we need 4x2 for m1 in matmul
            var m2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // 4x3
            CompareBackends("Addmm", false,
                be => { var aBase = CreateTensor(be, new long[] { 4, 2 }, new float[] { 1, 5, 2, 6, 3, 7, 4, 8 }); var a = aBase.Transpose(); var b = CreateTensor(be, new long[] { 4, 3 }, m2); var src = CreateTensor(be, new long[] { 2, 3 }, new float[6]); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Addmm(r, 0f, src, 1f, a, b); aBase.Dispose(); b.Dispose(); src.Dispose(); return r; },
                be => { var aBase = CreateTensor(be, new long[] { 4, 2 }, new float[] { 1, 5, 2, 6, 3, 7, 4, 8 }); var a = aBase.Transpose(); var b = CreateTensor(be, new long[] { 4, 3 }, m2); var src = CreateTensor(be, new long[] { 2, 3 }, new float[6]); var r = CreateTensor(be, new long[] { 2, 3 }, null); Ops.Addmm(r, 0f, src, 1f, a, b); aBase.Dispose(); b.Dispose(); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_AddmmBatch_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            // m1 (batch, rows, k) = (2, 3, 2), m2 (batch, k, cols) = (2, 2, 4), result (2, 3, 4)
            var m1 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // 2 x 3x2
            var m2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 2 x 2x4
            CompareBackends("AddmmBatch", true,
                be => { var a = CreateTensor(be, new long[] { 2, 3, 2 }, m1); var b = CreateTensor(be, new long[] { 2, 2, 4 }, m2); var src = CreateTensor(be, new long[] { 2, 3, 4 }, new float[24]); var r = CreateTensor(be, new long[] { 2, 3, 4 }, null); Ops.AddmmBatch(r, 0f, src, 1f, a, b); a.Dispose(); b.Dispose(); src.Dispose(); return r; },
                be => { var a = CreateTensor(be, new long[] { 2, 3, 2 }, m1); var b = CreateTensor(be, new long[] { 2, 2, 4 }, m2); var src = CreateTensor(be, new long[] { 2, 3, 4 }, new float[24]); var r = CreateTensor(be, new long[] { 2, 3, 4 }, null); Ops.AddmmBatch(r, 0f, src, 1f, a, b); a.Dispose(); b.Dispose(); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_AddmmBatch_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            // m1 (2,3,2) stored as (2,2,3) then transpose(1,2) -> (2,3,2) for non-contiguous
            var m1Base = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // 2x2x3 layout
            var m2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 2 x 2x4
            CompareBackends("AddmmBatch", false,
                be => { var aBase = CreateTensor(be, new long[] { 2, 2, 3 }, m1Base); var a = aBase.Transpose(1, 2); var b = CreateTensor(be, new long[] { 2, 2, 4 }, m2); var src = CreateTensor(be, new long[] { 2, 3, 4 }, new float[24]); var r = CreateTensor(be, new long[] { 2, 3, 4 }, null); Ops.AddmmBatch(r, 0f, src, 1f, a, b); aBase.Dispose(); b.Dispose(); src.Dispose(); return r; },
                be => { var aBase = CreateTensor(be, new long[] { 2, 2, 3 }, m1Base); var a = aBase.Transpose(1, 2); var b = CreateTensor(be, new long[] { 2, 2, 4 }, m2); var src = CreateTensor(be, new long[] { 2, 3, 4 }, new float[24]); var r = CreateTensor(be, new long[] { 2, 3, 4 }, null); Ops.AddmmBatch(r, 0f, src, 1f, a, b); aBase.Dispose(); b.Dispose(); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Sum_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Sum", true,
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Sum(r, src, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Sum(r, src, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Sum_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Sum", false,
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Sum(r, view, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Sum(r, view, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Mean_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Mean", true,
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Mean(r, src, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 3 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Mean(r, src, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Mean_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6 };
            CompareBackends("Mean", false,
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Mean(r, view, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 3, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Mean(r, view, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Argmax_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 3, 1, 4, 1, 5, 9, 2, 6 };
            CompareBackends("Argmax", true,
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Argmax(r, src, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Argmax(r, src, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_Argmax_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 3, 2, 1, 4, 5, 6, 9, 8 };
            CompareBackends("Argmax", false,
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Argmax(r, view, 1); src.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var r = CreateTensor(be, new long[] { 2, 1 }, null); Ops.Argmax(r, view, 1); src.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_LayerNorm_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var gamma = new float[] { 1, 1, 1, 1 };
            var beta = new float[] { 0, 0, 0, 0 };
            CompareBackends("LayerNorm", true,
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.LayerNorm(r, src, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.LayerNorm(r, src, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_LayerNorm_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 5, 2, 6, 3, 7, 4, 8 };
            var gamma = new float[] { 1, 1, 1, 1 };
            var beta = new float[] { 0, 0, 0, 0 };
            CompareBackends("LayerNorm", false,
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.LayerNorm(r, view, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.LayerNorm(r, view, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_RMSNorm_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var gamma = new float[] { 1, 1, 1, 1 };
            var beta = new float[] { 0, 0, 0, 0 };
            CompareBackends("RMSNorm", true,
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.RMSNorm(r, src, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 2, 4 }, vals); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.RMSNorm(r, src, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_RMSNorm_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var vals = new float[] { 1, 5, 2, 6, 3, 7, 4, 8 };
            var gamma = new float[] { 1, 1, 1, 1 };
            var beta = new float[] { 0, 0, 0, 0 };
            CompareBackends("RMSNorm", false,
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.RMSNorm(r, view, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; },
                be => { var src = CreateTensor(be, new long[] { 4, 2 }, vals); var view = src.Transpose(); var g = CreateTensor(be, new long[] { 4 }, gamma); var bt = CreateTensor(be, new long[] { 4 }, beta); var r = CreateTensor(be, new long[] { 2, 4 }, null); Ops.RMSNorm(r, view, g, bt, 1e-5f); src.Dispose(); g.Dispose(); bt.Dispose(); return r; });
        }

        [TestMethod]
        public void Ggml_AtomicAdd_Contiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var baseVals = new float[] { 1, 2, 3, 4, 5, 6 };
            var addVals = new float[] { 10, 20, 30, 40, 50, 60 };
            CompareBackends("AtomicAdd", true,
                be => { var result = CreateTensor(be, new long[] { 2, 3 }, baseVals); var rhs = CreateTensor(be, new long[] { 2, 3 }, addVals); Ops.AtomicAdd(result, rhs); return result; },
                be => { var result = CreateTensor(be, new long[] { 2, 3 }, baseVals); var rhs = CreateTensor(be, new long[] { 2, 3 }, addVals); Ops.AtomicAdd(result, rhs); return result; });
        }

        [TestMethod]
        public void Ggml_AtomicAdd_NonContiguous()
        {
            if (!IsMacOS) { Assert.Inconclusive("Requires macOS"); return; }
            var baseVals = new float[] { 1, 2, 3, 4, 5, 6 };
            var addVals = new float[] { 10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60 };
            CompareBackends("AtomicAdd", false,
                be => { var result = CreateTensor(be, new long[] { 2, 3 }, baseVals); var view = result.View(new long[] { 2, 1, 3 }).Expand(2, 2, 3); var rhs = CreateTensor(be, new long[] { 2, 2, 3 }, addVals); Ops.AtomicAdd(view, rhs); rhs.Dispose(); return Ops.AsContiguous(result); },
                be => { var result = CreateTensor(be, new long[] { 2, 3 }, baseVals); var view = result.View(new long[] { 2, 1, 3 }).Expand(2, 2, 3); var rhs = CreateTensor(be, new long[] { 2, 2, 3 }, addVals); Ops.AtomicAdd(view, rhs); rhs.Dispose(); return Ops.AsContiguous(result); });
        }
    }
}
