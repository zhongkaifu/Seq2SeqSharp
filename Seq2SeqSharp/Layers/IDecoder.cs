using Seq2SeqSharp.Tools;

namespace Seq2SeqSharp
{
    public interface IDecoder : INeuralUnit
    {
        void Reset(IWeightFactory weightFactory, int batchSize);
    }
}
