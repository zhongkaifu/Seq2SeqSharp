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
using Microsoft.Extensions.Logging.Abstractions;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class MultiProcessorNetworkWrapper<T> : IMultiProcessorNetworkWrapper where T : INeuralUnit
    {
        private readonly T[] m_networks;
        private readonly int m_defaultDeviceId;
    //    private readonly T m_networkOnDefaultDevice;
        private readonly bool m_isStaticWeights;
        private bool m_weightsSynced;

        private Dictionary<string, int> m_weightName2DefaultDeviceId = new Dictionary<string, int>();
        private Dictionary<int, T> m_deviceId2Network = new Dictionary<int, T>();

        public MultiProcessorNetworkWrapper(T networkOnDefaultDevice, int[] deviceIds, bool isStaticWeights = false)
        {
            m_networks = new T[deviceIds.Length];
            m_defaultDeviceId = networkOnDefaultDevice.GetDeviceId();
            //  m_networkOnDefaultDevice = networkOnDefaultDevice;
            m_isStaticWeights = isStaticWeights;
            m_weightsSynced = false;

            object locker = new object();
            Parallel.For(0, deviceIds.Length, i =>
            {
                if (deviceIds[i] == m_defaultDeviceId)
                {
                    m_networks[i] = networkOnDefaultDevice;
                }
                else
                {
                    m_networks[i] = (T)networkOnDefaultDevice.CloneToDeviceAt(deviceIds[i]);
                }

                lock (locker)
                {
                    m_deviceId2Network.Add(deviceIds[i], m_networks[i]);
                }
            });

            //for (int i = 0; i < deviceIds.Length; i++)
            //{
            //    if (deviceIds[i] == m_defaultDeviceId)
            //    {
            //        m_networks[i] = networkOnDefaultDevice;
            //    }
            //    else
            //    {
            //        m_networks[i] = (T)networkOnDefaultDevice.CloneToDeviceAt(deviceIds[i]);
            //    }

            //    m_deviceId2Network.Add(deviceIds[i], m_networks[i]);
            //}

            var raDeviceIds = new RoundArray<int>(deviceIds);
            var weights = networkOnDefaultDevice.GetParams();
            foreach (var weight in weights)
            {
                m_weightName2DefaultDeviceId.Add(weight.Name, raDeviceIds.GetNextItem());
            }
        }


        private IWeightTensor GetWeightFromNetwork(T network, string weightName)
        {
            var weights = network.GetParams();
            foreach (var weight in weights)
            {
                if (weight.Name == weightName)
                {
                    return weight;
                }
            }

            return null;
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


            foreach (var pair in m_weightName2DefaultDeviceId)
            {
                var weightName = pair.Key;
                int weightDefaultDeviceId = pair.Value;

                // Get weights on the default device
                IWeightTensor weightOnDefaultDevice = GetWeightFromNetwork(m_deviceId2Network[weightDefaultDeviceId], weightName);
                if (weightOnDefaultDevice == null)
                {
                    throw new NullReferenceException($"Weight '{weightName}' should be on its default device '{weightDefaultDeviceId}', but we didn't find it.");
                }

                Parallel.ForEach(m_deviceId2Network, pair =>
                {
                    try
                    {
                        int deviceId = pair.Key;
                        T network = pair.Value;
                        if (deviceId != weightDefaultDeviceId)
                        {
                            IWeightTensor weight = GetWeightFromNetwork(network, weightName);
                            weight.CopyWeightsFrom(weightOnDefaultDevice);
                        }

                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, $"Error Message = '{err.Message}'.");
                        Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                        throw;
                    }
                });
            }

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

            foreach (var pair in m_weightName2DefaultDeviceId)
            {
                var weightName = pair.Key;
                int weightDefaultDeviceId = pair.Value;

                // Get weights on the default device
                IWeightTensor weightOnDefaultDevice = GetWeightFromNetwork(m_deviceId2Network[weightDefaultDeviceId], weightName);
                if (weightOnDefaultDevice == null)
                {
                    throw new NullReferenceException($"Weight '{weightName}' should be on its default device '{weightDefaultDeviceId}', but we didn't find it.");
                }

                Parallel.ForEach(m_deviceId2Network, pair =>
                {
                    try
                    {
                        int deviceId = pair.Key;
                        T network = pair.Value;
                        if (deviceId != weightDefaultDeviceId)
                        {
                            IWeightTensor weight = GetWeightFromNetwork(network, weightName);
                            weightOnDefaultDevice.AddGradientFrom(weight);
                        }

                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, $"Error Message = '{err.Message}'.");
                        Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                        throw;
                    }
                });
            }
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

        public void Dispose()
        {
            Parallel.ForEach(m_networks, network =>
            {
                List<Tools.IWeightTensor> tensors = network.GetParams();
                for (int j = 0; j < tensors.Count; j++)
                {                 
                    tensors[j].ReleaseWeight();
                    tensors[j].ReleaseGradient();
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
                m_networks[0].Save(model);
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
                m_networks[0].Load(model);

                var srcWeights = m_networks[0].GetParams();
                for (int i = 1; i < m_networks.Length; i++)
                {
                    var destWeights = m_networks[i].GetParams();

                    for (int j = 0; j < srcWeights.Count; j++)
                    {
                        destWeights[j].CopyWeightsFrom(srcWeights[j]);
                    }
                }

                m_weightsSynced = true;
            }
        }


        public List<IWeightTensor> GetWeightsOnDefaultDevice()
        {
            List<IWeightTensor> weightsOnDeviceIds = new List<IWeightTensor>();
            foreach (var pair in m_weightName2DefaultDeviceId)
            {
                var weightName = pair.Key;
                int weightDefaultDeviceId = pair.Value;

                // Get weights on the default device
                IWeightTensor weightOnDefaultDevice = GetWeightFromNetwork(m_deviceId2Network[weightDefaultDeviceId], weightName);

                weightsOnDeviceIds.Add(weightOnDefaultDevice);
            }

            return weightsOnDeviceIds;
        }

        /// <summary>
        /// Return the network on specific device
        /// </summary>
        /// <param name="deviceIdIdx">The device id index. -1 is default device</param>
        /// <returns></returns>
        public T GetNetworkOnDevice(int deviceIdIdx)
        {
            //if (deviceIdIdx == -1)
            //{
            //    return m_networkOnDefaultDevice;
            //}
            return m_networks[deviceIdIdx];
        }
    }
}
