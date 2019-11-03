using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class MultiProcessorNetworkWrapper<T> : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        T[] m_networks;
        int m_defaultDeviceId;
        int[] m_deviceIds;
        T m_networkOnDefaultDevice;

        public MultiProcessorNetworkWrapper(T networkOnDefaultDevice, int[] deviceIds)
        {
            m_networks = new T[deviceIds.Length];
            m_defaultDeviceId = networkOnDefaultDevice.GetDeviceId();
            m_deviceIds = deviceIds;
            m_networkOnDefaultDevice = networkOnDefaultDevice;

            for (int i = 0; i < deviceIds.Length; i++)
            {
                if (deviceIds[i] == m_defaultDeviceId)
                {
                    m_networks[i] = networkOnDefaultDevice;
                }
                else
                {
                    m_networks[i] = (T)networkOnDefaultDevice.CloneToDeviceAt(deviceIds[i]);
                }
            }          
        }

        /// <summary>
        /// Copy weights from tensors on the default device to all other devices
        /// </summary>
        public void SyncWeights()
        {
            var tensorsOnDefaultDevice = m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(m_networks, network =>
            {
                if (network.Equals(m_networkOnDefaultDevice) == false)
                {
                    var tensors = network.GetParams();

                    for (int j = 0; j < tensors.Count; j++)
                    {
                        tensors[j].CopyWeightsFrom(tensorsOnDefaultDevice[j]);
                    }
                }

            });
        }

        /// <summary>
        /// Collect gradients from other devices and sum it up to the default device
        /// </summary>
        public void SumGradientsToNetworkOnDefaultDevice()
        {
            var tensorsOnDefaultDevice = m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(m_networks, network =>
            {
                if (network.Equals(m_networkOnDefaultDevice) == false)
                {
                    var tensors = network.GetParams();

                    for (int j = 0; j < tensors.Count; j++)
                    {
                        tensorsOnDefaultDevice[j].AddGradientFrom(tensors[j]);
                    }
                }

            });

        }

        /// <summary>
        /// Fill zero to all gradients on all devices
        /// </summary>
        public void ZeroGradientsOnAllDevices()
        {
            Parallel.ForEach(m_networks, network =>
            {
                var tensors = network.GetParams();
                for (int j = 0; j < tensors.Count; j++)
                {
                    tensors[j].ZeroGradient();
                }
            });
        }

        public void ZeroGradientCache()
        {
            var tensors = m_networkOnDefaultDevice.GetParams();
            foreach (var tensor in tensors)
            {
                tensor.ZeroCache();
            }
        }

        /// <summary>
        /// Save weights of the network on default device to given stream
        /// </summary>
        /// <param name="stream"></param>
        public void Save(Stream stream)
        {
            m_networkOnDefaultDevice.Save(stream);
        }

        /// <summary>
        /// Load weights from given stream to the network on default device
        /// </summary>
        /// <param name="stream"></param>
        public void Load(Stream stream)
        {
            m_networkOnDefaultDevice.Load(stream);
        }

        public T GetNetworkOnDefaultDevice()
        {
            return m_networkOnDefaultDevice;
        }

        public INeuralUnit GetNeuralUnitOnDefaultDevice()
        {
            return GetNetworkOnDefaultDevice();
        }

        /// <summary>
        /// Return the network on specific device
        /// </summary>
        /// <param name="deviceIdIdx">The device id index. -1 is default device</param>
        /// <returns></returns>
        public T GetNetworkOnDevice(int deviceIdIdx)
        {
            if (deviceIdIdx == -1)
            {
                return m_networkOnDefaultDevice;
            }
            return m_networks[deviceIdIdx];
        }
    }
}
