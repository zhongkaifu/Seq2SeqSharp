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

namespace Seq2SeqSharp.LearningRate
{
    public class CosineDecayLearningRate : ILearningRate
    {
        int m_decaySteps;
        float m_startLearningRate;
        int m_warmupSteps;
        private int m_weightsUpdateCount = 0; // How many steps have been already done

        /// <summary>
        /// 
        /// </summary>
        /// <param name="startLearningRate">The starting learning rate after warm up</param>
        /// <param name="warmupSteps">The steps for warming up</param>
        /// <param name="decaySteps">The total steps needs to be run</param>
        /// <param name="alreadyUpdatedSteps">The steps have been finished</param>
        public CosineDecayLearningRate(float startLearningRate, int warmupSteps, int decaySteps, int alreadyUpdatedSteps)
        {
            m_decaySteps = decaySteps;
            m_startLearningRate = startLearningRate;
            m_warmupSteps = warmupSteps;
            m_weightsUpdateCount = alreadyUpdatedSteps;

            Logger.WriteLine(Logger.Level.debug, $"Creating cosine decay learning rate. StartLearningRate = '{startLearningRate}', WarmupSteps = '{warmupSteps}', WeightsUpdatesCount = '{alreadyUpdatedSteps}', DecaySteps = '{decaySteps}''");
        }

        public float GetCurrentLearningRate(int epoch)
        {
            m_weightsUpdateCount++;

            if (m_weightsUpdateCount < m_warmupSteps)
            {
                float completed_fraction = (float)m_weightsUpdateCount / (float)m_warmupSteps;
                return completed_fraction * m_startLearningRate;
            }
            else
            {
                var step = Math.Min(m_weightsUpdateCount, m_decaySteps);
                var cosine_decay = 0.5 * (1.0 + Math.Cos(Math.PI * (float)step / (float)m_decaySteps));
                return (float)(m_startLearningRate * cosine_decay);

            }
        }
    }
}
