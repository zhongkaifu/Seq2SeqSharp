using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    public class DeviceKernel
    {
        private readonly byte[] ptx;


        public DeviceKernel(byte[] ptx)
        {
            this.ptx = ptx;
        }


    }
}
