using System;
using TensorSharp.Cpu;

namespace TensorSharp
{
    public static class ConvolutionOps
    {
        private static void EnsureSupportedStorage(Tensor tensor)
        {
            if (tensor.Storage is CpuStorage)
            {
                return;
            }

            if (tensor.Storage is CudaStorage)
            {
                return;
            }

            throw new NotSupportedException($"Tensor storage '{tensor.Storage?.GetType().Name}' is not supported by convolution operators.");
        }

        public static void Conv2DForward(Tensor input, Tensor output, Tensor weight, Tensor bias, ConvolutionDesc2d cd)
        {
            EnsureSupportedStorage(input);
            var finputShape = SpatialConvolutionMM.FInputSize(input.Sizes, output.Sizes, cd);
            using Tensor finput = new Tensor(input.Allocator, input.ElementType, finputShape);
            OpRegistry.Invoke("conv2dforward", input, output, weight, bias, finput, cd);
        }

        public static void Conv2DBackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, ConvolutionDesc2d cd)
        {
            EnsureSupportedStorage(input);
            var fShape = SpatialConvolutionMM.FInputSize(input.Sizes, gradOutput.Sizes, cd);
            using Tensor finput = new Tensor(input.Allocator, input.ElementType, fShape);
            using Tensor fgradInput = new Tensor(input.Allocator, input.ElementType, fShape);
            OpRegistry.Invoke("conv2dbackwardinput", input, gradOutput, gradInput, weight, finput, fgradInput, cd);
        }

        public static void Conv2DBackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, ConvolutionDesc2d cd)
        {
            EnsureSupportedStorage(input);
            var fShape = SpatialConvolutionMM.FInputSize(input.Sizes, gradOutput.Sizes, cd);
            using Tensor finput = new Tensor(input.Allocator, input.ElementType, fShape);
            using Tensor fgradInput = new Tensor(input.Allocator, input.ElementType, fShape);
            OpRegistry.Invoke("conv2dbackwardfilter", input, gradOutput, gradWeight, gradBias, finput, fgradInput, cd);
        }
    }
}
