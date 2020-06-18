//using System;
//using System.Drawing;
//using System.Drawing.Imaging;

//namespace TensorSharp
//{
//    public static class TensorImageExtensions
//    {
//        /// <summary>
//        /// Converts a Tensor to a Bitmap. Elements of the tensor are assumed to be normalized in the range [0, 1]
//        /// The tensor must have one of the following structures:
//        ///  * 2D tensor - output is a 24bit BGR bitmap in greyscale
//        ///  * 3D tensor where first dimension has length 1 - output is 24bit BGR bitmap in greyscale
//        ///  * 3D tensor where first dimension has length 3 - output is 24bit BGR bitmap
//        ///  * 3D tensor where first dimension has length 4 - output is 32bit BGRA bitmap
//        ///  
//        /// 2D tensors must be in HW (height x width) order;
//        /// 3D tensors must be in CHW (channel x height x width) order.
//        /// </summary>
//        /// <param name="tensor"></param>
//        /// <returns></returns>
//        public static Bitmap ToBitmap(this Tensor tensor)
//        {
//            if (tensor.DimensionCount != 2 && tensor.DimensionCount != 3)
//            {
//                throw new InvalidOperationException("tensor must have 2 or 3 dimensions");
//            }

//            if (tensor.DimensionCount == 3 &&
//                (tensor.Sizes[0] != 1 && tensor.Sizes[0] != 3 && tensor.Sizes[0] != 4))
//            {
//                throw new InvalidOperationException("3D tensor's first dimension (color channels) must be of length 1, 3 or 4");
//            }

//            Tensor src;
//            if (tensor.DimensionCount == 2)
//            {
//                src = tensor.RepeatTensor(3, 1, 1);
//            }
//            else if (tensor.DimensionCount == 3 && tensor.Sizes[0] == 1)
//            {
//                src = tensor.RepeatTensor(3, 1, 1);
//            }
//            else
//            {
//                src = tensor.CopyRef();
//            }

//            Cpu.CpuAllocator cpuAllocator = new Cpu.CpuAllocator();
//            long bytesPerPixel = src.Sizes[0];

//            try
//            {
//                using (Tensor cpuFloatTensor = new Tensor(cpuAllocator, DType.Float32, src.Sizes))
//                using (Tensor permutedFloatTensor = cpuFloatTensor.Permute(1, 2, 0))
//                {
//                    Ops.Copy(cpuFloatTensor, src);
//                    Ops.Mul(cpuFloatTensor, cpuFloatTensor, 255);

//                    PixelFormat resultFormat = bytesPerPixel == 3 ? PixelFormat.Format24bppRgb : PixelFormat.Format32bppArgb;
//                    Bitmap result = new Bitmap((int)src.Sizes[2], (int)src.Sizes[1], resultFormat);



//                    BitmapData lockData = result.LockBits(
//                        new Rectangle(0, 0, result.Width, result.Height),
//                        ImageLockMode.WriteOnly,
//                        result.PixelFormat);

//                    long[] sizes = new long[] { result.Height, result.Width, bytesPerPixel };
//                    long[] strides = new long[] { lockData.Stride, bytesPerPixel, 1 };
//                    Tensor resultTensor = new Tensor(cpuAllocator, DType.UInt8, sizes, strides);

//                    // Re-order tensor and convert to bytes
//                    Ops.Copy(resultTensor, permutedFloatTensor);

//                    int byteLength = lockData.Stride * lockData.Height;
//                    resultTensor.Storage.CopyFromStorage(lockData.Scan0, resultTensor.StorageOffset, byteLength);

//                    result.UnlockBits(lockData);
//                    return result;
//                }
//            }
//            finally
//            {
//                src.Dispose();
//            }
//        }
//    }
//}
