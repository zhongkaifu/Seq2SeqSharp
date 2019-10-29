
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{

    [Serializable]
    public class Encoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public Encoder(string name, int hdim, int dim, int depth, int deviceId)
        {
            encoders.Add(new LSTMCell($"{name}.LSTM_0", hdim, dim, deviceId));

            for (int i = 1; i < depth; i++)
            {
                encoders.Add(new LSTMCell($"{name}.LSTM_{i}", hdim, hdim, deviceId));

            }
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in encoders)
            {
                item.Reset(weightFactory, batchSize);
            }

        }

        public IWeightTensor Encode(IWeightTensor V, IComputeGraph g)
        {
            foreach (var encoder in encoders)
            {
                var e = encoder.Step(V, g);
                V = e;
            }

            return V;
        }


        public List<IWeightTensor> getParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (var item in encoders)
            {
                response.AddRange(item.getParams());

            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Load(stream);
            }
        }
    }
}
