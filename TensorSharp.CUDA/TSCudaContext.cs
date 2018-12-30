using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    public struct ScratchSpace
    {
        public int size;
        public CUdeviceptr buffer;
    }

    [Serializable]
    public class TSCudaContext : IDisposable
    {
        public const int MaxDims = 25;
        private const string CacheDir = @"cuda_cache\general";


        private readonly int deviceCount;
        private readonly DeviceState[] devices;
        private readonly bool[,] p2pAccess;

        private readonly RuntimeCompiler.KernelDiskCache diskCache;

        private readonly RuntimeCompiler.CudaCompiler compiler;
        private readonly CudaKernelCache kernelCache = new CudaKernelCache();
        

        public TSCudaContext()
        {
            try
            {
                this.deviceCount = CudaContext.GetDeviceCount();
            }
            catch
            {
                // CudaContext.GetDeviceCount() throws if CUDA drivers are not installed
                this.deviceCount = 0;
            }

            this.devices = Enumerable.Repeat(0, deviceCount)
                .Select(x => new DeviceState(x))
                .ToArray();

            if (deviceCount > 0)
            {
                p2pAccess = EnablePeerAccess(devices.Select(x => x.CudaContext).ToArray(), devices[0].CudaContext);
            }
            else
            {
                p2pAccess = new bool[0, 0];
            }

            this.diskCache = new RuntimeCompiler.KernelDiskCache(Path.Combine(Environment.CurrentDirectory, CacheDir));
            this.compiler = new RuntimeCompiler.CudaCompiler(diskCache);

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public RuntimeCompiler.CudaCompiler Compiler { get { return compiler; } }
        public CudaKernelCache KernelCache { get { return kernelCache; } }


        public void FreeMemory()
        {
            foreach (var device in devices)
            {
                device.FreeMemory();
            }
        }

        public void Dispose()
        {
            kernelCache.Dispose();

            foreach (var device in devices)
            {
                device.Dispose();
            }
        }

        public void Synchronize(int deviceId)
        {
            devices[deviceId].CudaContext.Synchronize();
        }

        public void SynchronizeAll()
        {
            foreach (var device in devices)
            {
                device.CudaContext.Synchronize();
            }
        }

        public CudaContext CudaContextForDevice(int deviceId)
        {
            return devices[deviceId].CudaContext;
        }

        public IDeviceAllocator AllocatorForDevice(int deviceId)
        {
            return devices[deviceId].MemoryAllocator;
        }

        public CudaContext CudaContextForTensor(Tensor tensor)
        {
            return CudaContextForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public ScratchSpace ScratchSpaceForDevice(int deviceId)
        {
            return devices[deviceId].ScratchSpace;
        }

        public PooledObject<CudaBlas> BlasForDevice(int deviceId)
        {
            return devices[deviceId].BlasHandles.Get();
        }

        public PooledObject<CudaBlas> BlasForTensor(Tensor tensor)
        {
            return BlasForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public bool CanAccessPeer(int srcDevice, int peerDevice)
        {
            return p2pAccess[srcDevice, peerDevice];
        }

        public CudaDeviceProperties DeviceInfoForContext(CudaContext cudaContext)
        {
            return devices[cudaContext.DeviceId].DeviceInfo;
        }

        

        // Returns a matrix of [i, j] values where [i, j] is true iff device i can access device j
        private static bool[,] EnablePeerAccess(CudaContext[] cudaContexts, CudaContext restoreCurrent)
        {
            var result = new bool[cudaContexts.Length, cudaContexts.Length];

            for (int i = 0; i < cudaContexts.Length; ++i)
            {
                for (int j = 0; j < cudaContexts.Length; ++j)
                {
                    if (i == j)
                    {
                        result[i, j] = true;
                    }
                    else
                    {
                        result[i, j] = EnablePeers(cudaContexts[i], cudaContexts[j]);
                    }
                }
            }

            restoreCurrent.SetCurrent();
            return result;
        }

        private static bool EnablePeers(CudaContext src, CudaContext target)
        {
            if (!src.DeviceCanAccessPeer(target))
                return false;

            src.SetCurrent();

            try
            {
                CudaContext.EnablePeerAccess(target);
                return true;
            }
            catch
            {
                return false;
            }
        }


        public void Precompile()
        {
            Precompile(Console.Write);
        }

        public void Precompile(Action<string> precompileProgressWriter)
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach (var applyType in assembly.TypesWithAttribute<PrecompileAttribute>(true).Where(x => !x.Item1.IsAbstract))
            {
                precompileProgressWriter("Precompiling " + applyType.Item1.Name + "\n");

                var instance = (IPrecompilable)Activator.CreateInstance(applyType.Item1);
                instance.Precompile(Compiler);
            }
        }

        public void CleanUnusedPTX()
        {
            diskCache.CleanUnused();
        }
    }
}
