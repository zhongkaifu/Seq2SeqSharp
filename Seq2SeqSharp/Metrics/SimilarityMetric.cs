using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Metrics
{
    public class SimilarityMetric : IMetric
    {
        public string Name => "Similarity_Metric";


        private float totalSimScore = 0.0f;
        private int totalSimScoreNum = 0;

        public void ClearStatus()
        {
            totalSimScore = 0.0f;
            totalSimScoreNum = 0;
        }

        public void Evaluate(List<List<string>> refTokens, List<string> hypTokens)
        {
            try
            {
                for (int i = 0; i < hypTokens.Count; i++)
                {
                    float hypSimScore = float.Parse(hypTokens[i]);
                    float refSimScore = float.Parse(refTokens[0][i]);

                    totalSimScore = Math.Abs(hypSimScore - refSimScore);
                    totalSimScoreNum++;
                }
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Exception: {err.Message}, Ref = '{String.Join(" ", refTokens)}', Hyp = '{String.Join(" ", hypTokens)}'");
                throw err;

            }
        }

        public double GetPrimaryScore()
        {
            return (1.0 - totalSimScore) / totalSimScoreNum;
        }

        public string GetScoreStr()
        {
            return $"Similarity Score = '{GetPrimaryScore()}'";
        }
    }
}
