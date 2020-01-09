using System.Collections.Generic;

namespace Seq2SeqSharp.Metrics
{
    public class LengthRatioMetric : IMetric
    {
        private double[] m_counts;
        public string Name => "Length Ratio (Hyp:Ref)";

        public LengthRatioMetric()
        {
            ClearStatus();
        }

        public void ClearStatus()
        {
            m_counts = new double[2];
        }

        public void Evaluate(List<List<string>> refTokens, List<string> hypTokens)
        {
            m_counts[0] += hypTokens.Count;
            m_counts[1] += BleuMetric.GetClosestRefLength(refTokens, hypTokens);
        }

        public string GetScoreStr()
        {
            return GetPrimaryScore().ToString("F");
        }

        public double GetPrimaryScore()
        {
            double lr = m_counts[0] / m_counts[1];
            return 100.0 * lr;
        }
    }
}
