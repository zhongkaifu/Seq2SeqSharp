using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightFactory
    {
        IWeightTensor CreateWeights(int row, int column, int deviceId, bool cleanWeights, string name = "", bool isTrainable = false);

        void Clear();
    }
}
