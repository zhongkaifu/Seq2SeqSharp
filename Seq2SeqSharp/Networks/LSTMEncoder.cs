// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{

    [Serializable]
    public class LSTMEncoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int Hdim { get; set; }
        public int Dim { get; set; }
        public int Depth { get; set; }

        public LSTMEncoder(string name, int hdim, int dim, int depth, int deviceId, bool isTrainable)
        {
            encoders.Add(new LSTMCell($"{name}.LSTM_0", hdim, dim, deviceId, isTrainable));

            for (int i = 1; i < depth; i++)
            {
                encoders.Add(new LSTMCell($"{name}.LSTM_{i}", hdim, hdim, deviceId, isTrainable));

            }
            this.Hdim = hdim;
            this.Dim = dim;
            this.Depth = depth;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (LSTMCell item in encoders)
            {
                item.Reset(weightFactory, batchSize);
            }

        }

        public IWeightTensor Encode(IWeightTensor V, IComputeGraph g)
        {
            foreach (LSTMCell encoder in encoders)
            {
                IWeightTensor e = encoder.Step(V, g);
                V = e;
            }

            return V;
        }


        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (LSTMCell item in encoders)
            {
                response.AddRange(item.getParams());

            }

            return response;
        }

        public void Save(IModel stream)
        {
            foreach (LSTMCell item in encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(IModel stream)
        {
            foreach (LSTMCell item in encoders)
            {
                item.Load(stream);
            }
        }
    }
}
