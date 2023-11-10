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
    public class DecayLearningRate : ILearningRate
    {
        private readonly float m_startLearningRate = 0.001f;
        private int m_weightsUpdateCount = 0;
        private readonly int m_warmupSteps = 8000;
        private readonly float m_stepDownFactor = 1.0f;
        private readonly int m_updateNumToStepDownLearningRate = 0;

        public DecayLearningRate(float startLearningRate, int warmupSteps, int weightsUpdatesCount, float stepDownFactor = 1.0f, int updateNumToStepDownLearningRate = 0)
        {
            if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                Logger.WriteLine($"Creating decay learning rate. StartLearningRate = '{startLearningRate}', WarmupSteps = '{warmupSteps}', WeightsUpdatesCount = '{weightsUpdatesCount}', StepDownFactor = '{stepDownFactor}', UpdateNumToStepDownLearningRate = '{updateNumToStepDownLearningRate}'");

            if (stepDownFactor != 1.0f)
            {
                if (updateNumToStepDownLearningRate > 0)
                {
                    if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                        Logger.WriteLine($"Step down learning rateo to '{stepDownFactor}' * current_learning_rate after every '{updateNumToStepDownLearningRate}' updates.");
                }
                else
                {
                    if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                        Logger.WriteLine($"Step down learning rateo to '{stepDownFactor}' * current_learning_rate after each epoch.");
                }
            }


            m_startLearningRate = startLearningRate;
            m_warmupSteps = warmupSteps;
            m_weightsUpdateCount = weightsUpdatesCount;
            m_stepDownFactor = stepDownFactor;
            m_updateNumToStepDownLearningRate = updateNumToStepDownLearningRate;
        }

        public float GetCurrentLearningRate(int epoch)
        {
            m_weightsUpdateCount++;

            int stepDownFactor = epoch;
            if (m_updateNumToStepDownLearningRate > 0)
            {
                stepDownFactor = m_weightsUpdateCount / m_updateNumToStepDownLearningRate;
            }

            float lr = m_startLearningRate * (float)(Math.Pow(m_stepDownFactor, stepDownFactor) * Math.Min(Math.Pow(m_weightsUpdateCount, -0.5), Math.Pow(m_warmupSteps, -1.5) * m_weightsUpdateCount) / Math.Pow(m_warmupSteps, -0.5));
            return lr;
        }
    }
}
