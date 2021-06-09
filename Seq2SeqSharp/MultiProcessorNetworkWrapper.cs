using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class MultiProcessorNetworkWrapper<T> : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        private readonly T[] m_networks;
        private readonly int m_defaultDeviceId;
        private readonly int[] m_deviceIds;
        private readonly T m_networkOnDefaultDevice;
        private readonly bool m_isStaticWeights;
        private bool m_weightsSynced;

        public MultiProcessorNetworkWrapper(T networkOnDefaultDevice, int[] deviceIds, bool isStaticWeights = false)
        {
            m_networks = new T[deviceIds.Length];
            m_defaultDeviceId = networkOnDefaultDevice.GetDeviceId();
            m_deviceIds = deviceIds;
            m_networkOnDefaultDevice = networkOnDefaultDevice;
            m_isStaticWeights = isStaticWeights;
            m_weightsSynced = false;

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
            if (m_isStaticWeights && m_weightsSynced)
            {
                return;
            }

            List<Tools.IWeightTensor> tensorsOnDefaultDevice = m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(m_networks, network =>
            {
                if (network.Equals(m_networkOnDefaultDevice) == false)
                {
                    List<Tools.IWeightTensor> tensors = network.GetParams();

                    for (int j = 0; j < tensors.Count; j++)
                    {
                        tensors[j].CopyWeightsFrom(tensorsOnDefaultDevice[j]);
                    }
                }

            });

            m_weightsSynced = true;
        }

        /// <summary>
        /// Collect gradients from other devices and sum it up to the default device
        /// </summary>
        public void SumGradientsToNetworkOnDefaultDevice()
        {
            if (m_isStaticWeights)
            {
                return;
            }

            List<Tools.IWeightTensor> tensorsOnDefaultDevice = m_networkOnDefaultDevice.GetParams();
            Parallel.ForEach(m_networks, network =>
            {
                if (network.Equals(m_networkOnDefaultDevice) == false)
                {
                    List<Tools.IWeightTensor> tensors = network.GetParams();

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
            if (m_isStaticWeights)
            {
                return;
            }

            Parallel.ForEach(m_networks, network =>
            {
                List<Tools.IWeightTensor> tensors = network.GetParams();
                for (int j = 0; j < tensors.Count; j++)
                {
                    tensors[j].ZeroGradient();
                }
            });
        }


        /// <summary>
        /// Release gradients on all devices
        /// </summary>
        public void ReleaseGradientsOnAllDevices()
        {
            if (m_isStaticWeights)
            {
                return;
            }

            Parallel.ForEach(m_networks, network =>
            {
                List<Tools.IWeightTensor> tensors = network.GetParams();
                for (int j = 0; j < tensors.Count; j++)
                {
                    tensors[j].ReleaseGradient();
                }
            });
        }

        /// <summary>
        /// Save weights of the network on default device to given stream
        /// </summary>
        /// <param name="stream"></param>
        public void Save(IModelMetaData stream)
        {
            if (m_isStaticWeights == false)
            {
                m_networkOnDefaultDevice.Save(stream);
            }
        }

        /// <summary>
        /// Load weights from given stream to the network on default device
        /// </summary>
        /// <param name="stream"></param>
        public void Load(IModelMetaData stream)
        {
            if (m_isStaticWeights == false)
            {
                m_networkOnDefaultDevice.Load(stream);
            }
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
