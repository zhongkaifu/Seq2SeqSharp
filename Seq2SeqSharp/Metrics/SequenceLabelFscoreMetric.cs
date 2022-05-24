using AdvUtils;
using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public class SequenceLabelFscoreMetric : IMetric
    {
        private double[] m_count;
        private readonly string m_classLabel;

        public string Name => $"SequenceLabelFscore ({m_classLabel})";

        public SequenceLabelFscoreMetric(string classLabel)
        {
            m_count = new double[3];
            m_classLabel = classLabel;

            Logger.WriteLine($"Creating sequence label F1 score metric for '{classLabel}'");
        }

        public void ClearStatus()
        {
            m_count = new double[3];
        }

        public void Evaluate(List<List<string>> allRefTokens, List<string> hypTokens)
        {
            foreach (List<string> refTokens in allRefTokens)
            {
                for (int i = 0; i < hypTokens.Count; i++)
                {
                    if (hypTokens[i] == m_classLabel)
                    {
                        m_count[1]++;
                    }
                    if (refTokens[i] == m_classLabel)
                    {
                        m_count[2]++;
                    }
                    if (hypTokens[i] == m_classLabel && refTokens[i] == m_classLabel)
                    {
                        m_count[0]++;
                    }
                }
            }
        }

        public string GetScoreStr()
        {
            if (m_count[1] == 0.0 || m_count[2] == 0.0)
            {
                return $"No F-score available for '{m_classLabel}'";
            }

            double precision = m_count[0] / m_count[1];
            double recall = m_count[0] / m_count[2];
            double objective = 0.0;
            if (precision > 0.0 && recall > 0.0)
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return $"F-score = '{100.0 * objective:F}' Precision = '{100.0 * precision:F}' Recall = '{100.0 * recall:F}'";
        }

        public double GetPrimaryScore()
        {
            if (m_count[1] == 0.0 || m_count[2] == 0.0)
            {
                return 0.0;
            }

            double precision = m_count[0] / m_count[1];
            double recall = m_count[0] / m_count[2];
            double objective = 0.0;
            if (precision > 0.0 && recall > 0.0)
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return 100.0 * objective;
        }
    }
}
