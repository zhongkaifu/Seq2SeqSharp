using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Seq2SeqSharp.Tools;
using System.IO;

namespace Seq2SeqSharp
{
    public interface IEncoder : INeuralUnit
    {
        IWeightTensor Encode(IWeightTensor rawInput, int batchSize, IComputeGraph g);
        void Reset(IWeightFactory weightFactory, int batchSize);
    }
}
