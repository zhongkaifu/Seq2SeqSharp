//using System;
//using System.Drawing;
//using System.Drawing.Imaging;

//namespace TensorSharp
//{
//    public static class BitmapExtensions
//    {
//        /// <summary>
//        /// Returns a Tensor constructed from the data in the Bitmap. The Tensor's dimensions are
//        /// ordered CHW (channel x height x width). The color channel dimension is in the same order
//        /// as the original Bitmap data. For 24bit bitmaps, this will be BGR. For 32bit bitmaps this
//        /// will be BGRA.
//        /// </summary>
//        /// <param name="bitmap"></param>
//        /// <param name="allocator"></param>
//        /// <returns></returns>
//        public static Tensor ToTensor(this Bitmap bitmap, IAllocator allocator)
//        {
//            Cpu.CpuAllocator cpuAllocator = new Cpu.CpuAllocator();

//            int bytesPerPixel = 0;

//            if (bitmap.PixelFormat == PixelFormat.Format24bppRgb)
//            {
//                bytesPerPixel = 3;
//            }
//            else if (bitmap.PixelFormat == PixelFormat.Format32bppArgb ||
//                bitmap.PixelFormat == PixelFormat.Format32bppPArgb ||
//                bitmap.PixelFormat == PixelFormat.Format32bppRgb)
//            {
//                bytesPerPixel = 4;
//            }
//            else
//            {
//                throw new InvalidOperationException("Bitmap must be 24bit or 32bit");
//            }

//            BitmapData lockData = bitmap.LockBits(
//                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
//                ImageLockMode.ReadOnly,
//                bitmap.PixelFormat);

//            try
//            {

//                long[] sizes = new long[] { bitmap.Height, bitmap.Width, bytesPerPixel };
//                long[] strides = new long[] { lockData.Stride, bytesPerPixel, 1 };
//                using (Tensor cpuByteTensor = new Tensor(cpuAllocator, DType.UInt8, sizes, strides))
//                {
//                    cpuByteTensor.Storage.CopyToStorage(cpuByteTensor.StorageOffset, lockData.Scan0, cpuByteTensor.Storage.ByteLength);
//                    using (Tensor permutedTensor = cpuByteTensor.Permute(2, 0, 1))
//                    {
//                        using (Tensor cpuFloatTensor = new Tensor(cpuAllocator, DType.Float32, permutedTensor.Sizes))
//                        {
//                            Ops.Copy(cpuFloatTensor, permutedTensor);

//                            // TODO this could be made more efficient by skipping a the following copy if allocator is a CpuAllocator,
//                            // but make sure that in that case the result tensor is not disposed before returning.

//                            Tensor result = new Tensor(allocator, DType.Float32, permutedTensor.Sizes);
//                            Ops.Copy(result, cpuFloatTensor);
//                            Ops.Div(result, result, 255);
//                            return result;
//                        }
//                    }
//                }
//            }
//            finally
//            {
//                bitmap.UnlockBits(lockData);
//            }

//        }
//    }
//}
