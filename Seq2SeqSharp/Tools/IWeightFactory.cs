using System;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightFactory : IDisposable
    {
        WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, IComputeGraph graphToBind = null);
    }
}
