using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IModel
    {
        public void AddWeights(string name, float[] weights);

        public float[] GetWeights(string name);

        public void ClearWeights();
    }
}
