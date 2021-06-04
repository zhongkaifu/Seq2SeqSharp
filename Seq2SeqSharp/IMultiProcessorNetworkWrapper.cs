using System.IO;

namespace Seq2SeqSharp
{
    public interface IMultiProcessorNetworkWrapper
    {
        void Save(Stream stream);
        void Load(Stream stream);
        void SyncWeights();
        void SumGradientsToNetworkOnDefaultDevice();
        INeuralUnit GetNeuralUnitOnDefaultDevice();
        void ZeroGradientsOnAllDevices();
        void ReleaseGradientsOnAllDevices();
    }
}
