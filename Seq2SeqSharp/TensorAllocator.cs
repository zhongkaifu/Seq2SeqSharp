using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{
    public static class TensorAllocator
    {
        private static IAllocator allocator = null;
        public static IAllocator Allocator
        {
            get
            {
                if (allocator == null)
                {
                    var cudaContext = new TSCudaContext();
                    cudaContext.Precompile(Console.Write);
                    cudaContext.CleanUnusedPTX();
                    allocator = new CudaAllocator(cudaContext, 0);
                }

                return allocator;
            }
        }
    }
}
