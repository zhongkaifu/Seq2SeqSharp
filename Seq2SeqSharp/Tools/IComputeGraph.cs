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
        IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, int batchSize, float alpha = 1.0f);

        IWeightTensor Mul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Add(IWeightTensor w1, IWeightTensor w2);
        IWeightTensor Tanh(IWeightTensor w, bool updateWeightsInPlace = false);
        IWeightTensor Sigmoid(IWeightTensor w, bool updateWeightsInPlace = false);
        IWeightTensor Relu(IWeightTensor w);

        IWeightTensor BuildPositionMatrix(int row, int column);
        IWeightTensor MulAdd(IWeightTensor m1, IWeightTensor m2, IWeightTensor m3);

        IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4);

        List<IWeightTensor> UnFolderRow(IWeightTensor m, int n, bool gradient = true);

        IWeightTensor PermuteBatch(IWeightTensor m, int batchSize);

        IWeightTensor Permute(IWeightTensor w, params int[] dims);

        IWeightTensor View(IWeightTensor w, params long[] dims);

        IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2);

        IWeightTensor ConcatColumns(IWeightTensor m1, IWeightTensor m2);

        void Backward();
        void RunTopBackward();

        IWeightTensor PeekRow(IWeightTensor w, int ix, int num = 1, bool runGradients = true);
        IWeightTensor Dropout(IWeightTensor V, float drop_prob);

        IWeightTensor Softmax(IWeightTensor w, bool bp = true);

        IWeightTensor ConcatColumns(params IWeightTensor[] wl);        

        List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes);
        (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2);
        (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3);

        IWeightTensor ConcatRows(List<IWeightTensor> wl);

        IWeightTensor RepeatRows(IWeightTensor w, int n);

        IWeightTensor Transpose(IWeightTensor w);
		
		 IWeightTensor Mul(IWeightTensor w, float v);

        IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f);
    }
}
