using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class CostEventArg : EventArgs
    {
        public float LearningRate { get; set; }
        public float Cost { get; set; }
        public double CostInTotal { get; set; }

        public int Epoch { get; set; }

        public int ProcessedInTotal { get; set; }

        public int SentenceLength { get; set; }

        public DateTime StartDateTime { get; set; }
    }
}
