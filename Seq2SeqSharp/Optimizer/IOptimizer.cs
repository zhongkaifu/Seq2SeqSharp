using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Text;

namespace Seq2SeqSharp.Optimizer
{
    public interface IOptimizer
    {
        void UpdateWeights(List<IWeightTensor> model, int batchSize, float step_size, float regc, int iter);
    }
}
