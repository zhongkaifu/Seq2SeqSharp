using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    public interface INeuralUnit
    {
        List<IWeightTensor> GetParams();
        void Save(Stream stream);
        void Load(Stream stream);

        INeuralUnit CloneToDeviceAt(int deviceId);
        int GetDeviceId();
    }
}
