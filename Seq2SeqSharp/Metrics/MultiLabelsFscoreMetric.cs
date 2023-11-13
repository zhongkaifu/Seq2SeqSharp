using AdvUtils;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public class MultiLabelsFscoreMetric : IMetric
    {
        private List<double[]> m_counts;
        private readonly List<string> m_classLabels;

        public string Name => $"MultiLabelsFscore_{m_groupName}";
        private readonly string m_groupName;

        public MultiLabelsFscoreMetric(string groupName, List<string> classLabels)
        {
            m_classLabels = classLabels;
            m_groupName = groupName;
            ClearStatus();

            Logger.WriteLine(Logger.Level.debug, $"Added '{string.Join(" ", classLabels)}' labels to '{Name}'.");
        }

        public void ClearStatus()
        {
            m_counts = new List<double[]>();
            for (int i = 0; i < m_classLabels.Count; i++)
            {
                m_counts.Add(new double[3]);
            }
        }

        public void Evaluate(List<List<string>> allRefTokens, List<string> hypTokens)
        {
            for (int j = 0; j < m_counts.Count; j++)
            {
                var m_count = m_counts[j];
                foreach (List<string> refTokens in allRefTokens)
                {
                    try
                    {
                        for (int i = 0; i < hypTokens.Count; i++)
                        {
                            if (hypTokens[i] == m_classLabels[j])
                            {
                                m_count[1]++;
                            }
                            if (refTokens[i] == m_classLabels[j])
                            {
                                m_count[2]++;
                            }
                            if (hypTokens[i] == m_classLabels[j] && refTokens[i] == m_classLabels[j])
                            {
                                m_count[0]++;
                            }
                        }
                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, $"Exception: {err.Message}, Ref = '{string.Join(" ", refTokens)}', Hyp = '{string.Join(" ", hypTokens)}'");
                        throw;

                    }
                }
            }
        }

        public string GetScoreStr()
        {
            List<string> results = new List<string>();
            for (int i = 0; i < m_counts.Count; i++)
            {
                var m_count = m_counts[i];
                if (m_count[1] == 0.0 || m_count[2] == 0.0)
                {
                    continue;
                }

                double precision = m_count[0] / m_count[1];
                double recall = m_count[0] / m_count[2];
                double objective = 0.0;
                if (precision > 0.0 && recall > 0.0)
                {
                    objective = 2.0 * (precision * recall) / (precision + recall);
                }

                results.Add($"'{m_classLabels[i]}': F-score = '{100.0 * objective:F}' Precision = '{100.0 * precision:F}' Recall = '{100.0 * recall:F}'");
            }

            return string.Join("\n", results) + $"\nThe number of categories = '{m_counts.Count}'\n";
        }

        public double GetPrimaryScore()
        {
            double score = 0.0;
            for (int i = 0; i < m_counts.Count; i++)
            {
                var m_count = m_counts[i];
                if (m_count[1] == 0.0 || m_count[2] == 0.0)
                {
                    score += 0.0;
                    continue;
                }

                double precision = m_count[0] / m_count[1];
                double recall = m_count[0] / m_count[2];
                double objective = 0.0;
                if (precision > 0.0 && recall > 0.0)
                {
                    objective = 2.0 * (precision * recall) / (precision + recall);
                }

                score += 100.0 * objective;
            }

            return score / m_counts.Count;
        }
    }
}
