using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class CostEventArg : EventArgs
    {
        public float CostPerWord { get; set; }
        public double AvgCostInTotal { get; set; }

        public int Epoch { get; set; }
        public int Update { get; set; }

        public int ProcessedSentencesInTotal { get; set; }

        public long ProcessedWordsInTotal { get; set; }

        public DateTime StartDateTime { get; set; }

        public float LearningRate { get; set; }

        public int BatchSize;
    }
}
