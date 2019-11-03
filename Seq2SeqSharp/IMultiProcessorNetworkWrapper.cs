using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        void ZeroGradientCache();
    }
}
