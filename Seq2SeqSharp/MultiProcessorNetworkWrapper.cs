// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class MultiProcessorNetworkWrapper<T> : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        private readonly T[] m_networks;
        private readonly int m_defaultDeviceId;
        private readonly T m_networkOnDefaultDevice;
        private readonly bool m_isStaticWeights;
        private bool m_weightsSynced;

        public MultiProcessorNetworkWrapper(T networkOnDefaultDevice, int[] deviceIds, bool isStaticWeights = false)
        {
            m_networks = new T[deviceIds.Length];
            m_defaultDeviceId = networkOnDefaultDevice.GetDeviceId();
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
                try
                {
                    if (network.Equals(m_networkOnDefaultDevice) == false)
                    {
                        List<Tools.IWeightTensor> tensors = network.GetParams();

                        for (int j = 0; j < tensors.Count; j++)
                        {
                            tensors[j].CopyWeightsFrom(tensorsOnDefaultDevice[j]);
                        }
                    }
                }
                catch (Exception err)
                {
                    Logger.WriteLine(Logger.Level.err, $"Error Message = '{err.Message}', Call Stack = '{err.StackTrace}'");
                    throw;
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
        /// Save weights of the network on default device to given model
        /// </summary>
        /// <param name="model"></param>
        public void Save(IModel model)
        {
            if (m_isStaticWeights == false)
            {
                m_networkOnDefaultDevice.Save(model);
            }
        }

        /// <summary>
        /// Load weights from given model to networks on all devices
        /// </summary>
        /// <param name="model"></param>
        public void Load(IModel model)
        {
            if (m_isStaticWeights == false)
            {
                for (int i = 0; i < m_networks.Length; i++)
                {
                    m_networks[i].Load(model);
                }
                m_weightsSynced = true;
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
