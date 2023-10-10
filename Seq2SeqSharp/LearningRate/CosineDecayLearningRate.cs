using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.LearningRate
{
    public class CosineDecayLearningRate : ILearningRate
    {
        float m_initial_learning_rate;
        int m_decaySteps;
        float m_warmup_target;
        int m_warmupSteps;
        float m_initial_decay_lr;
        private int m_weightsUpdateCount = 0;
        float m_alpha = 0.0f;

        public CosineDecayLearningRate(float startLearningRate, int warmupSteps, int decaySteps, int weightsUpdateCount)
        {
            m_initial_learning_rate = 0.0f;
            m_decaySteps = decaySteps;
            m_warmup_target = startLearningRate;
            m_warmupSteps = warmupSteps;

            if (m_warmup_target == 0)
            {
                m_initial_decay_lr = m_initial_learning_rate;
            }
            else
            {
                m_initial_decay_lr = m_warmup_target;
            }

            m_weightsUpdateCount = weightsUpdateCount;

            Logger.WriteLine($"Creating cosine decay learning rate. StartLearningRate = '{startLearningRate}', WarmupSteps = '{warmupSteps}', WeightsUpdatesCount = '{weightsUpdateCount}', DecaySteps = '{decaySteps}''");
        }

        public float GetCurrentLearningRate(int epoch)
        {
            m_weightsUpdateCount++;

            if (m_weightsUpdateCount < m_warmupSteps)
            {
                float completed_fraction = (float)m_weightsUpdateCount / (float)m_warmupSteps;
                float total_delta = m_warmup_target - m_initial_learning_rate;

                return completed_fraction * total_delta;
            }
            else
            {
                var step = Math.Min(m_weightsUpdateCount, m_decaySteps);
                var cosine_decay = 0.5 * (1.0 + Math.Cos(Math.PI * (float)step / (float)m_decaySteps));
                var decayed = (1.0 - m_alpha) * cosine_decay + m_alpha;

                return (float)(m_initial_decay_lr * decayed);

            }
        }
    }
}
