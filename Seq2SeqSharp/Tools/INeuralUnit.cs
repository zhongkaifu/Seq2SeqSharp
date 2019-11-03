using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
