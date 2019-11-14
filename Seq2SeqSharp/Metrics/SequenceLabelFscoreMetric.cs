using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Metrics
{
    public class SequenceLabelFscoreMetric : IMetric
    {
        private double[] m_count;
        private string m_classLabel;

        public string Name => $"SequenceLabelFscore ({m_classLabel})";

        public SequenceLabelFscoreMetric(string classLabel)
        {
            m_count = new double[3];
            m_classLabel = classLabel;
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
            double precision = m_count[0] / m_count[1];
            double recall = m_count[0] / m_count[2];
            double objective = 0.0;
            if (precision > 0.0 && recall > 0.0)
            {
                objective = 2.0 * (precision * recall) / (precision + recall);
            }

            return $"F-score = '{(100.0 * objective).ToString("F")}' Precision = '{(100.0 * precision).ToString("F")}' Recall = '{(100.0 * recall).ToString("F")}'";
        }

        public double GetPrimaryScore()
        {
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
