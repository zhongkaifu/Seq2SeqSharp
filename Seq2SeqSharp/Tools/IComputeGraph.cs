using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public interface IComputeGraph
    {
        IWeightMatrix Mul(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Mul(SparseWeightMatrix m1, IWeightMatrix w2);
        IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4);
        IWeightMatrix Tanh(IWeightMatrix w);
        IWeightMatrix Sigmoid(IWeightMatrix w);

        IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3);

        IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2);

        IWeightMatrix ConcatColumns(IWeightMatrix m1, IWeightMatrix m2);
        void Backward();
        IWeightMatrix PeekRow(IWeightMatrix w, int ix);
        IWeightMatrix Dropout(IWeightMatrix V, float drop_prob);
        IWeightMatrix SoftmaxWithCrossEntropy(IWeightMatrix src);

        void DropoutPredict(IWeightMatrix V, float drop_prob);

        IWeightMatrix Softmax(IWeightMatrix w);
        IWeightMatrix ConcatColumns(IWeightMatrix[] wl);        

        List<IWeightMatrix> SplitColumns(IWeightMatrix w, params int[] sizes);

        IWeightMatrix ConcatRows(List<IWeightMatrix> wl);

        IWeightMatrix RepeatRows(IWeightMatrix w, int n);

        IWeightMatrix Transpose2(IWeightMatrix w);

        List<IWeightMatrix> SplitRows(IWeightMatrix w, params int[] sizes);

    }
}
