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
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;

namespace Seq2SeqSharp
{

    [Serializable]
    public class BiEncoder : IEncoder
    {
        private readonly List<LSTMCell> m_forwardEncoders;
        private readonly List<LSTMCell> m_backwardEncoders;
        private readonly string m_name;
        private readonly int m_hiddenDim;
        private readonly int m_inputDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;

        public BiEncoder(string name, int hiddenDim, int inputDim, int depth, int deviceId, bool isTrainable, DType elementType = DType.Float32)
        {
            if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                Logger.WriteLine($"Creating BiLSTM encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', IsTrainable = '{isTrainable}'");

            m_forwardEncoders = new List<LSTMCell>();
            m_backwardEncoders = new List<LSTMCell>();

            for (int i = 0; i < depth; i++)
            {
                m_forwardEncoders.Add(new LSTMCell($"{name}.Forward_LSTM_{i}", hiddenDim, hiddenDim, deviceId, isTrainable: isTrainable, elementType: elementType));
                m_backwardEncoders.Add(new LSTMCell($"{name}.Backward_LSTM_{i}", hiddenDim, hiddenDim, deviceId, isTrainable: isTrainable, elementType: elementType));
            }

            m_name = name;
            m_hiddenDim = hiddenDim;
            m_inputDim = inputDim;
            m_depth = depth;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new BiEncoder(m_name, m_hiddenDim, m_inputDim, m_depth, deviceId, m_isTrainable);
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (LSTMCell item in m_forwardEncoders)
            {
                item.Reset(weightFactory, batchSize);
            }

            foreach (LSTMCell item in m_backwardEncoders)
            {
                item.Reset(weightFactory, batchSize);
            }
        }

        public IWeightTensor Encode(IWeightTensor rawInputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            int seqLen = rawInputs.Rows / batchSize;

            rawInputs = g.TransposeBatch(rawInputs, seqLen);

            List<IWeightTensor> inputs = new List<IWeightTensor>();
            for (int i = 0; i < seqLen; i++)
            {
                IWeightTensor emb_i = g.Peek(rawInputs, 0, i * batchSize, batchSize);
                inputs.Add(emb_i);
            }

            List<IWeightTensor> forwardOutputs = new List<IWeightTensor>();
            List<IWeightTensor> backwardOutputs = new List<IWeightTensor>();

            List<IWeightTensor> layerOutputs = inputs.ToList();
            for (int i = 0; i < m_depth; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    IWeightTensor forwardOutput = m_forwardEncoders[i].Step(layerOutputs[j], g);
                    forwardOutputs.Add(forwardOutput);

                    IWeightTensor backwardOutput = m_backwardEncoders[i].Step(layerOutputs[inputs.Count - j - 1], g);
                    backwardOutputs.Add(backwardOutput);
                }

                backwardOutputs.Reverse();
                layerOutputs.Clear();
                for (int j = 0; j < seqLen; j++)
                {
                    IWeightTensor addW = g.Add(forwardOutputs[j], backwardOutputs[j]);
                    layerOutputs.Add(addW);
                }

            }

            var result = g.Concate(layerOutputs, 0);

            return g.TransposeBatch(result, batchSize);
        }


        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (LSTMCell item in m_forwardEncoders)
            {
                response.AddRange(item.getParams());
            }


            foreach (LSTMCell item in m_backwardEncoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(IModel stream)
        {
            foreach (LSTMCell item in m_forwardEncoders)
            {
                item.Save(stream);
            }

            foreach (LSTMCell item in m_backwardEncoders)
            {
                item.Save(stream);
            }
        }

        public void Load(IModel stream)
        {
            foreach (LSTMCell item in m_forwardEncoders)
            {
                item.Load(stream);
            }

            foreach (LSTMCell item in m_backwardEncoders)
            {
                item.Load(stream);
            }
        }
    }
}
