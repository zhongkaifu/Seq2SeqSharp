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
