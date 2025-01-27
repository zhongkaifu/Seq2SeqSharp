using Seq2SeqSharp.Metrics;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public class CostEventArg : EventArgs
    {
        public double AvgCostInTotal { get; set; }

        public double AvgChosenRewardInTotal { get; set; }
        public double AvgRejectedRewardInTotal { get; set; }

        public int Epoch { get; set; }
        public int Update { get; set; }

        public int ProcessedSentencesInTotal { get; set; }

        public long ProcessedWordsInTotal { get; set; }

        public DateTime StartDateTime { get; set; }

        public float LearningRate { get; set; }

        public float LossScaling { get; set; }
    }

    public class EvaluationEventArg : EventArgs
    {
        public string Message;
        public ConsoleColor Color;
        public string Title;
        public List<IMetric> Metrics;
        public bool BetterModel;
    }

    public class KVCacheRemoveEventArg : EventArgs
    {
        public string Key;
        public object Value;
        public string Reason;
    }
}
