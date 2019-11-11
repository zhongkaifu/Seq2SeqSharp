using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Metrics
{
    public interface IMetric
    {
        void Evaluate(List<List<string>> refTokens, List<string> hypTokens);
        double GetScore();
        string Name { get; }
        void ClearStatus();

    }
}
