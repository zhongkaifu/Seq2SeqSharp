using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Core
{
    public static class TensorConcatenation
    {
        // Requires an implementation of *copy* for the given tensor types
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            if (inputs.Length < 2) throw new ArgumentException("Concat: at least two tensors required", "inputs");

            var ndim = Math.Max(dimension, inputs.Max(x => x.DimensionCount));
            var size = ConcatTensorSize(ndim, dimension, inputs);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, inputs[0], false, size);


            // Select each region of the result corresponding to each input tensor,
            // and copy into the result
            long offset = 0;
            for (int j = 0; j < inputs.Length; ++j)
            {
                var dimSize = GetDimSize(inputs[j], dimension);
                using (var nt = writeTarget.Narrow(dimension, offset, dimSize))
                {
                    Ops.Copy(nt, inputs[j]);
                }
                offset += dimSize;
            }

            return writeTarget;
        }



        private static long GetDimSize(Tensor tensor, int dim)
        {
            return dim < tensor.DimensionCount ? tensor.Sizes[dim] : 1;
        }

        private static long[] ConcatTensorSize(int ndim, int dimension, Tensor[] tensors)
        {
            var result = new long[ndim];
            for (int i = 0; i < ndim; ++i)
            {
                var dimSize = GetDimSize(tensors[0], i);
                if (i == dimension)
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        dimSize += GetDimSize(tensors[j], i);
                    }
                }
                else
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        if (dimSize != GetDimSize(tensors[j], i))
                        {
                            throw new InvalidOperationException("Inconsistent tensor sizes");
                        }
                    }
                }
                result[i] = dimSize;
            }
            return result;
        }

    }
}
