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
        public double avgCostInTotal { get; set; }

        public int Epoch { get; set; }

        public int ProcessedInTotal { get; set; }

        public DateTime StartDateTime { get; set; }

        public float AvgLearningRate { get; set; }
    }
}
