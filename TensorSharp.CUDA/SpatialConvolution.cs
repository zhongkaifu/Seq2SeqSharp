using System;
using TensorSharp.Cpu;
using TensorSharp.CUDA.DeviceCode;

namespace TensorSharp.CUDA
{
    public class SpatialConvolution
    {
        private readonly Im2ColKernels im2colKernels = new Im2ColKernels();

        public static long[] FInputSize(long[] inputSizes, long[] outputSizes, ConvolutionDesc2d cd)
        {
            return new long[] { cd.kW * cd.kH * inputSizes[1], outputSizes[2] * outputSizes[3] };
        }

        public void Conv2Forward(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput, ConvolutionDesc2d cd)
        {
            var geometry = ConvGeometry.Build(input, weight, cd);
            geometry.ValidateForwardBuffers(output, finput, bias);

            for (long batchIndex = 0; batchIndex < geometry.BatchSize; ++batchIndex)
            {
                using var inputSlice = input.Select(0, batchIndex);
                using var outputSlice = output.Select(0, batchIndex);
                using var output2d = outputSlice.View(geometry.OutputPlanes, geometry.OutputSpatialSize);

                if (bias != null)
                {
                    using var biasExp = bias.Expand(geometry.OutputPlanes, geometry.OutputSpatialSize);
                    Ops.Copy(output2d, biasExp);
                }
                else
                {
                    Ops.Fill(outputSlice, 0.0f);
                }

                im2colKernels.Im2Col(
                    inputSlice,
                    finput,
                    (int)geometry.InputPlanes,
                    (int)geometry.InputHeight,
                    (int)geometry.InputWidth,
                    geometry.KernelH,
                    geometry.KernelW,
                    geometry.PadH,
                    geometry.PadW,
                    geometry.StrideH,
                    geometry.StrideW,
                    1,
                    1);

                Ops.Addmm(output2d, 1.0f, output2d, 1.0f, weight, finput);
            }
        }

        public void Conv2BackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            var geometry = ConvGeometry.Build(input, weight, cd);
            geometry.ValidateBackwardInputBuffers(gradOutput, gradInput, fgradInput);

            using var weightT = weight.Transpose();

            for (long batchIndex = 0; batchIndex < geometry.BatchSize; ++batchIndex)
            {
                using var gradInputSlice = gradInput.Select(0, batchIndex);
                using var gradOutputSlice = gradOutput.Select(0, batchIndex);
                using var gradOutput2d = gradOutputSlice.View(geometry.OutputPlanes, geometry.OutputSpatialSize);

                Ops.Addmm(fgradInput, 0.0f, fgradInput, 1.0f, weightT, gradOutput2d);

                im2colKernels.Col2Im(
                    fgradInput,
                    gradInputSlice,
                    (int)geometry.InputPlanes,
                    (int)geometry.InputHeight,
                    (int)geometry.InputWidth,
                    geometry.KernelH,
                    geometry.KernelW,
                    geometry.PadH,
                    geometry.PadW,
                    geometry.StrideH,
                    geometry.StrideW,
                    1,
                    1);
            }
        }

        public void Conv2BackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            var geometry = ConvGeometry.Build(input, gradWeight, cd);
            geometry.ValidateBackwardFilterBuffers(gradOutput, gradWeight, gradBias);

            for (long batchIndex = 0; batchIndex < geometry.BatchSize; ++batchIndex)
            {
                using var inputSlice = input.Select(0, batchIndex);
                using var gradOutputSlice = gradOutput.Select(0, batchIndex);

                im2colKernels.Im2Col(
                    inputSlice,
                    finput,
                    (int)geometry.InputPlanes,
                    (int)geometry.InputHeight,
                    (int)geometry.InputWidth,
                    geometry.KernelH,
                    geometry.KernelW,
                    geometry.PadH,
                    geometry.PadW,
                    geometry.StrideH,
                    geometry.StrideW,
                    1,
                    1);

                using var gradOutput2d = gradOutputSlice.View(geometry.OutputPlanes, geometry.OutputSpatialSize);
                using var finputT = finput.Transpose();

                Ops.Addmm(gradWeight, 1.0f, gradWeight, 1.0f, gradOutput2d, finputT);

                if (gradBias != null)
                {
                    Ops.Sum(gradBias, gradOutput2d, 1);
                }
            }
        }

        private readonly struct ConvGeometry
        {
            public long BatchSize { get; }
            public long InputPlanes { get; }
            public long InputHeight { get; }
            public long InputWidth { get; }
            public long OutputPlanes { get; }
            public long OutputHeight { get; }
            public long OutputWidth { get; }
            public int KernelH { get; }
            public int KernelW { get; }
            public int StrideH { get; }
            public int StrideW { get; }
            public int PadH { get; }
            public int PadW { get; }
            public long OutputSpatialSize => OutputHeight * OutputWidth;
            public long KernelSize => (long)KernelH * KernelW;

            private ConvGeometry(long batchSize, long inputPlanes, long inputHeight, long inputWidth, long outputPlanes, long outputHeight, long outputWidth, ConvolutionDesc2d cd)
            {
                BatchSize = batchSize;
                InputPlanes = inputPlanes;
                InputHeight = inputHeight;
                InputWidth = inputWidth;
                OutputPlanes = outputPlanes;
                OutputHeight = outputHeight;
                OutputWidth = outputWidth;
                KernelH = cd.kH;
                KernelW = cd.kW;
                StrideH = cd.dH;
                StrideW = cd.dW;
                PadH = cd.padH;
                PadW = cd.padW;
            }

            public static ConvGeometry Build(Tensor input, Tensor weight, ConvolutionDesc2d cd)
            {
                if (input.DimensionCount != 4)
                {
                    throw new InvalidOperationException("4D input expected (NCHW order)");
                }

                var batch = input.Sizes[0];
                var inPlane = input.Sizes[1];
                var inHeight = input.Sizes[2];
                var inWidth = input.Sizes[3];
                var outPlane = weight.Sizes[0];
                var kernelInput = weight.Sizes[1];
                var expectedKernelInput = (long)cd.kW * cd.kH * inPlane;

                if (kernelInput != expectedKernelInput)
                {
                    throw new InvalidOperationException(
                        $"Input has incorrect number of channels. Got {kernelInput / (cd.kW * cd.kH)} input planes, expected {inPlane}.");
                }

                var outputWidth = (inWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
                var outputHeight = (inHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

                if (outputWidth < 1 || outputHeight < 1)
                {
                    throw new InvalidOperationException(
                        $"Output size too small; calculated output size = ({outPlane}x{outputHeight}x{outputWidth})");
                }

                return new ConvGeometry(batch, inPlane, inHeight, inWidth, outPlane, outputHeight, outputWidth, cd);
            }

            public void ValidateForwardBuffers(Tensor output, Tensor finput, Tensor bias)
            {
                if (output.Sizes[0] != BatchSize ||
                    output.Sizes[1] != OutputPlanes ||
                    output.Sizes[2] != OutputHeight ||
                    output.Sizes[3] != OutputWidth)
                {
                    throw new InvalidOperationException("output is incorrect size");
                }

                if (finput.Sizes[0] != KernelSize * InputPlanes ||
                    finput.Sizes[1] != OutputSpatialSize)
                {
                    throw new InvalidOperationException("finput is incorrect size");
                }

                if (bias != null && bias.Sizes[0] != OutputPlanes)
                {
                    throw new InvalidOperationException("bias has incorrect size");
                }
            }

            public void ValidateBackwardInputBuffers(Tensor gradOutput, Tensor gradInput, Tensor fgradInput)
            {
                if (gradOutput.Sizes[0] != BatchSize ||
                    gradOutput.Sizes[1] != OutputPlanes ||
                    gradOutput.Sizes[2] != OutputHeight ||
                    gradOutput.Sizes[3] != OutputWidth)
                {
                    throw new InvalidOperationException("gradOutput is incorrect size");
                }

                if (gradInput.Sizes[0] != BatchSize ||
                    gradInput.Sizes[1] != InputPlanes ||
                    gradInput.Sizes[2] != InputHeight ||
                    gradInput.Sizes[3] != InputWidth)
                {
                    throw new InvalidOperationException("gradInput is incorrect size");
                }

                if (fgradInput.Sizes[0] != KernelSize * InputPlanes ||
                    fgradInput.Sizes[1] != OutputSpatialSize)
                {
                    throw new InvalidOperationException("fgradInput is incorrect size");
                }
            }

            public void ValidateBackwardFilterBuffers(Tensor gradOutput, Tensor gradWeight, Tensor gradBias)
            {
                if (gradOutput.Sizes[0] != BatchSize ||
                    gradOutput.Sizes[1] != OutputPlanes ||
                    gradOutput.Sizes[2] != OutputHeight ||
                    gradOutput.Sizes[3] != OutputWidth)
                {
                    throw new InvalidOperationException("gradOutput is incorrect size");
                }

                if (gradWeight.Sizes[0] != OutputPlanes ||
                    gradWeight.Sizes[1] != KernelSize * InputPlanes)
                {
                    throw new InvalidOperationException("gradWeight is incorrect size");
                }

                if (gradBias != null && gradBias.Sizes[0] != OutputPlanes)
                {
                    throw new InvalidOperationException("gradBias is incorrect size");
                }
            }
        }
    }

    [OpsClass]
    public class CudaConvolutionOps
    {
        private static readonly object locker = new object();
        private static readonly SpatialConvolution cudaConv = new SpatialConvolution();

        [RegisterOpStorageType("conv2dforward", typeof(CudaStorage))]
        public void Conv2DForward(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput, ConvolutionDesc2d cd)
        {
            lock (locker)
            {
                cudaConv.Conv2Forward(input, output, weight, bias, finput, cd);
            }
        }

        [RegisterOpStorageType("conv2dbackwardinput", typeof(CudaStorage))]
        public void Conv2DBackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            lock (locker)
            {
                cudaConv.Conv2BackwardInput(input, gradOutput, gradInput, weight, finput, fgradInput, cd);
            }
        }

        [RegisterOpStorageType("conv2dbackwardfilter", typeof(CudaStorage))]
        public void Conv2DBackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            lock (locker)
            {
                cudaConv.Conv2BackwardFilter(input, gradOutput, gradWeight, gradBias, finput, fgradInput, cd);
            }
        }
    }
}
