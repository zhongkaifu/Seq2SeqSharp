using System;

namespace Seq2SeqSharp
{
    public class CostEventArg : EventArgs
    {
        public double AvgCostInTotal { get; set; }

        public int Epoch { get; set; }
        public int Update { get; set; }

        public int ProcessedSentencesInTotal { get; set; }

        public long ProcessedWordsInTotal { get; set; }

        public DateTime StartDateTime { get; set; }

        public float LearningRate { get; set; }
    }
}
