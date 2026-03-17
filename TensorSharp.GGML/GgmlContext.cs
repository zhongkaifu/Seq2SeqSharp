using System;
using System.Reflection;

namespace TensorSharp.GGML
{
    public sealed class GgmlContext
    {
        internal GgmlMemoryPool MemoryPool { get; }

        public GgmlContext(int[] deviceIds)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            if (deviceIds.Length != 1)
            {
                throw new NotSupportedException("The GGML Metal backend currently supports a single device only.");
            }

            DeviceId = deviceIds[0];
            MemoryPool = new GgmlMemoryPool();
            MemoryPool.EnsureInitialBlocks();
            GgmlNative.EnsureAvailable();
            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public int DeviceId { get; }
    }
}
