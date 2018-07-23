using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public interface IComputeGraph
    {
        WeightMatrix mul(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix mul(SparseWeightMatrix m1, WeightMatrix m2);
        WeightMatrix addsigmoid(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3);
        WeightMatrix addtanh(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4);
        WeightMatrix addtanh(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4, WeightMatrix m5);
        WeightMatrix addtanh(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4, WeightMatrix m5, WeightMatrix m6);
        WeightMatrix addtanh(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3);
        WeightMatrix eltmul(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix add(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix tanh(WeightMatrix m);

        WeightMatrix addsigmoid(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4);
        WeightMatrix addsigmoid(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4, WeightMatrix m5);
        WeightMatrix addsigmoid(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3, WeightMatrix m4, WeightMatrix m5, WeightMatrix m6);
        WeightMatrix muladd(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3);
        WeightMatrix addtanh(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix Softmax(WeightMatrix m);
        List<WeightMatrix> Softmax(WeightMatrix[] m);

        WeightMatrix scalemul(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix scalemuladd(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3);
        WeightMatrix RepeatRows(WeightMatrix m1, int rows);
        WeightMatrix concatColumns(WeightMatrix m1, WeightMatrix m2);
        WeightMatrix weightRows(WeightMatrix m1, WeightMatrix weightRow);
        WeightMatrix sumColumns(WeightMatrix m1);
        void backward();
        WeightMatrix PeekRow(WeightMatrix m, int ix);
        WeightMatrix Dropout(WeightMatrix V, float drop_prob);
        WeightMatrix SoftmaxWithCrossEntropy(WeightMatrix m);

    }
}
