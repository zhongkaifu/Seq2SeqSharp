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
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IMultiProcessorNetworkWrapper
    {
        void Save(IModel model);
        void Load(IModel model);
        void SyncWeights();
        void SumGradientsToNetworkOnDefaultDevice();
        List<IWeightTensor> GetWeightsOnDefaultDevice();
        void ZeroGradientsOnAllDevices();
        void ReleaseGradientsOnAllDevices();

        void Dispose();
    }
}
