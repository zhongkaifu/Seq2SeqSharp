using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Layers
{
    public interface IFeedForwardLayer : INeuralUnit
    {
        IWeightTensor Process(IWeightTensor inputT, int batchSize, IComputeGraph g);

    }
}
