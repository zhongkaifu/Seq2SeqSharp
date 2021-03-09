using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp.Utils
{
    public class TensorUtils
    {
        //public static IWeightTensor BuildTensorFrom2DArray(List<List<int>> array, int deviceId, params long[] shape)
        //{
        //    float[] buf = new float[array.Count * array[0].Count];
        //    Array.Fill(buf, 0.0f);


        //    for (int i = 0; i < array.Count; i++)
        //    {
        //        for (int j = 0; j < array[0].Count; j++)
        //        {
        //            buf[i * array[0].Count + j] = array[i][j];
        //        }
        //    }

        //    WeightTensor tensor = new WeightTensor(shape, deviceId, $"BuildTensorFrom2DArray_{deviceId}", isTrainable: false);
        //    tensor.SetWeightArray(buf);

        //    return tensor;
        //}

        public static void Scatter(IWeightTensor res, IWeightTensor source, IWeightTensor indices, int dim)
        {
            WeightTensor i = indices as WeightTensor;
            WeightTensor s = source as WeightTensor;
            WeightTensor r = res as WeightTensor;

            Ops.Scatter(r.TWeight, s.TWeight, dim, i.TWeight);
        }

        //public static IWeightTensor Gather(IWeightTensor src, IWeightTensor indices, int dim, int deviceId)
        //{
        //    WeightTensor i = indices as WeightTensor;
        //    WeightTensor s = src as WeightTensor;


        //    WeightTensor res = new WeightTensor(indices.Sizes, deviceId, name: $"Gather_{deviceId}", isTrainable: false);
        //    Ops.Gather(res.TWeight, s.TWeight, dim, i.TWeight);

        //    return res;
        //}
    }
}
