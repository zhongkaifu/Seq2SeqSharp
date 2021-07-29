using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public enum RefHypIdx
    {
        RefIdx = 0,
        HypIdx = 1
    }

    public class BleuMetric : IMetric
    {
        private double[] m_counts;
        private readonly int m_matchIndex;

        private bool CaseInsensitive { get; }
        private int NgramOrder { get; }
        public string Name => "BLEU";

        public BleuMetric(int ngramOrder = 4, bool caseInsensitive = true)
        {
            NgramOrder = ngramOrder;
            CaseInsensitive = caseInsensitive;
            m_matchIndex = (int)RefHypIdx.HypIdx + NgramOrder;

            ClearStatus();
        }

        public void ClearStatus()
        {
            m_counts = new double[1 + 2 * NgramOrder];
        }

        public void Evaluate(List<List<string>> refTokens, List<string> hypTokens)
        {
            if (CaseInsensitive)
            {
                for (int i = 0; i < refTokens.Count; i++)
                {
                    refTokens[i] = ToLowerCase(refTokens[i]);
                }
                hypTokens = ToLowerCase(hypTokens);
            }
            List<Dictionary<string, int>> refCounts = new List<Dictionary<string, int>>();
            for (int n = 0; n < NgramOrder; n++)
            {
                refCounts.Add(new Dictionary<string, int>());
            }
            foreach (List<string> r in refTokens)
            {
                List<Dictionary<string, int>> counts = GetNgramCounts(r);
                for (int n = 0; n < NgramOrder; n++)
                {
                    foreach (KeyValuePair<string, int> e in counts[n])
                    {
                        string ngram = e.Key;
                        int count = e.Value;
                        if (!refCounts[n].ContainsKey(ngram) || count > refCounts[n][ngram])
                        {
                            refCounts[n][ngram] = count;
                        }
                    }
                }
            }

            List<Dictionary<string, int>> hypCounts = GetNgramCounts(hypTokens);

            m_counts[(int)RefHypIdx.RefIdx] += GetClosestRefLength(refTokens, hypTokens);
            for (int j = 0; j < NgramOrder; j++)
            {
                int overlap = 0;
                foreach (KeyValuePair<string, int> e in hypCounts[j])
                {
                    string ngram = e.Key;
                    int hypCount = e.Value;
                    if (refCounts[j].TryGetValue(ngram, out int refCount))
                    {
                        overlap += Math.Min(hypCount, refCount);
                    }
                }
                m_counts[(int)RefHypIdx.HypIdx + j] += Math.Max(0, hypTokens.Count - j);
                m_counts[m_matchIndex + j] += overlap;
            }
        }

        public string GetScoreStr()
        {
            return GetPrimaryScore().ToString("F");
        }

        public double GetPrimaryScore()
        {
            double precision = Precision();
            double bp = BrevityPenalty();

            return 100.0 * precision * bp;
        }

        internal static double GetClosestRefLength(List<List<string>> refTokens, List<string> hypTokens)
        {
            int closestIndex = -1;
            int closestDistance = int.MaxValue;
            for (int i = 0; i < refTokens.Count; i++)
            {
                int distance = Math.Abs(refTokens[i].Count - hypTokens.Count);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestIndex = i;
                }
            }

            return refTokens[closestIndex].Count;
        }

        private double BrevityPenalty()
        {
            double refLen = m_counts[(int)RefHypIdx.RefIdx];
            double hypLen = m_counts[(int)RefHypIdx.HypIdx];
            if (hypLen == 0.0 || hypLen >= refLen)
            {
                return 1.0;
            }

            return Math.Exp(1.0 - refLen / hypLen);
        }

        private List<Dictionary<string, int>> GetNgramCounts(List<string> tokens)
        {
            List<Dictionary<string, int>> allCounts = new List<Dictionary<string, int>>();
            for (int n = 0; n < NgramOrder; n++)
            {
                Dictionary<string, int> counts = new Dictionary<string, int>();
                for (int i = 0; i < tokens.Count - n; i++)
                {
                    string ngram = string.Join(" ", tokens.ToArray(), i, n + 1);
                    if (!counts.ContainsKey(ngram))
                    {
                        counts[ngram] = 1;
                    }
                    else
                    {
                        counts[ngram]++;
                    }
                }
                allCounts.Add(counts);
            }
            return allCounts;
        }

        private double Precision()
        {
            double prec = 1.0;
            for (int i = 0; i < NgramOrder; i++)
            {
                double x = m_counts[m_matchIndex + i] / (m_counts[(int)RefHypIdx.HypIdx + i] + 0.001);
                prec *= Math.Pow(x, 1.0 / NgramOrder);
            }
            return prec;
        }

        private static List<string> ToLowerCase(List<string> tokens)
        {
            List<string> output = new List<string>();
            for (int i = 0; i < tokens.Count; i++)
            {
                output.Add(tokens[i].ToLower());
            }
            return output;
        }
    }
}
