// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA
{
    // Represents a compile-time specialization of ApplyN.
    // If all tensors are small enough, the kernel will use 32-bit indices
    // The kernels are also specialized for contiguous tensors, tensors with a
    // small number of dimensions, and a totally generic 'specialization'.
    // If TensorDims[i] == -2, then tensor i is entirely contiguous
    // If TensorDims[i] == -1, a totally generic kernel should be generated for that tensor.
    public class ApplySpecialization
    {
        public const string IndexType32 = "unsigned __int32";
        public const string IndexType64 = "unsigned __int64";


        public bool Use32BitIndices { get; private set; }
        public int[] TensorDims { get; private set; }

        public DType[] TensorElementTypes { get;private set; }

        public ApplySpecialization(params Tensor[] tensors)
        {
            if (tensors.All(ApplyUtils.CanUse32BitIndexMath))
            {
                Use32BitIndices = true;

                // Specialize each tensor dimenionality independently
                TensorDims = tensors.Select(tensor =>
                {
                    if (tensor.IsContiguous())
                    {
                        return -2;
                    }

                    return -1; // tensor.DimensionCount > 3 ? -1 : tensor.DimensionCount;
                })
                .ToArray();
            }
            else
            {
                Use32BitIndices = false;
                // For 64-bit index case (ie. large tensors), only specalize on totally contiguous
                // or totally generic
                if (tensors.All(x => x.IsContiguous()))
                {
                    // All tensors are contiguous
                    TensorDims = Enumerable.Repeat(-2, tensors.Length).ToArray();
                }
                else
                {
                    // Not all tensors are contiguous - just generate a completely generic kernel
                    TensorDims = Enumerable.Repeat(-1, tensors.Length).ToArray();
                }
            }

            TensorElementTypes = new DType[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
            {
                TensorElementTypes[i] = tensors[i].ElementType;
            }
        }

        public ApplySpecialization(bool use32BitIndices, DType[] elementTypes, params int[] tensorDims)
        {
            Use32BitIndices = use32BitIndices;
            TensorDims = tensorDims;
            TensorElementTypes = elementTypes;
        }



        public KernelConfig GetConfig()
        {
            KernelConfig result = new KernelConfig();

            result.Set("INDEX_TYPE", Use32BitIndices ? IndexType32 : IndexType64);

            for (int i = 0; i < TensorDims.Length; ++i)
            {
                char tensorName = (char)('A' + i);
                result.Set("DIMS" + tensorName, TensorDims[i].ToString());
            }

            return result;
        }

        public static IEnumerable<ApplySpecialization> AllSpecializations(int tensorCount, DType[] elementTypes = null)
        {
            if (elementTypes == null)
            {
                elementTypes = new DType[tensorCount];
                for (int i = 0; i < tensorCount; i++)
                {
                    elementTypes[i] = DType.Float32;
                }
            }
            else
            {
                if (tensorCount != elementTypes.Length)
                {
                    throw new ArgumentException($"Inconsistent tensor count '{tensorCount}' and the number of elementTypes '{elementTypes.Length}'");
                }
            }


            yield return new ApplySpecialization(false, elementTypes, Enumerable.Repeat(-2, tensorCount).ToArray());
            yield return new ApplySpecialization(false, elementTypes, Enumerable.Repeat(-1, tensorCount).ToArray());

            foreach (int[] combination in CombinationsOf(All32BitTensorDims, tensorCount))
            {
                yield return new ApplySpecialization(true, elementTypes, combination);
            }
        }

        private static readonly int[] All32BitTensorDims = new int[] { -2, -1 }; //, 1, 2, 3 };

        private static IEnumerable<T[]> CombinationsOf<T>(T[] possibleValues, int count)
        {
            if (count < 1)
            {
                throw new ArgumentOutOfRangeException("count");
            }

            if (count == 1)
            {
                foreach (T item in possibleValues)
                {
                    yield return new T[] { item };
                }
            }
            else
            {
                foreach (T item in possibleValues)
                {
                    IEnumerable<T[]> restCombinations = CombinationsOf(possibleValues, count - 1);
                    foreach (T[] restItems in restCombinations)
                    {
                        List<T> result = new List<T>(count);
                        result.AddRange(restItems);
                        result.Add(item);
                        yield return result.ToArray();
                    }
                }
            }
        }
    }
}
