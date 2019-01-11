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
        IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4);
        IWeightMatrix Tanh(IWeightMatrix w);
        IWeightMatrix Sigmoid(IWeightMatrix w);

        IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3);

        IWeightMatrix MulAdd2(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3);

        List<IWeightMatrix> UnFolderRow(IWeightMatrix m, int n, bool gradient = true);

        IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2);

        IWeightMatrix ConcatColumns(IWeightMatrix m1, IWeightMatrix m2);

        void Backward();
        IWeightMatrix PeekRow(IWeightMatrix w, int ix, int num = 1);
        IWeightMatrix Dropout(IWeightMatrix V, float drop_prob);
        IWeightMatrix SoftmaxWithCrossEntropy(IWeightMatrix src);

        IWeightMatrix Softmax(IWeightMatrix w);

        IWeightMatrix SoftmaxM(IWeightMatrix w, bool bp = true);

        IWeightMatrix ConcatColumns(params IWeightMatrix[] wl);        

        List<IWeightMatrix> SplitColumns2(IWeightMatrix w, params int[] sizes);
        (IWeightMatrix r1, IWeightMatrix r2) SplitColumns(IWeightMatrix w, int size1, int size2);
        (IWeightMatrix r1, IWeightMatrix r2, IWeightMatrix r3) SplitColumns(IWeightMatrix w, int size1, int size2, int size3);

        IWeightMatrix ConcatRows(List<IWeightMatrix> wl, bool bp = true);

        IWeightMatrix RepeatRows(IWeightMatrix w, int n);

        IWeightMatrix Transpose2(IWeightMatrix w);
    }
}
