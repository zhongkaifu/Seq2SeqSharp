// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using TensorSharp;

namespace Seq2SeqSharp.Utils
{
    public enum PositionEmbeddingEnums
    {
        APE,
        NoPE,
        RoPE,
    }

    public class PositionEmbedding
    {

        public static IWeightTensor AddPositionEmbedding(IComputeGraph g, IWeightTensor posEmbedding, int batchSize, IWeightTensor inputEmbs, float dropoutRatio)
        {
            var Column = posEmbedding.Columns;
            int seqLen = inputEmbs.Rows / batchSize;

            IWeightTensor posEmbeddingPeek = g.Peek(posEmbedding, 0, 0, seqLen);
            using (var posEmbeddingPeekView = g.View(posEmbeddingPeek, dims: new long[] { 1, seqLen, Column }))
            {
                using (var posEmbeddingPeekViewExp = g.Expand(posEmbeddingPeekView, dims: new long[] { batchSize, seqLen, Column }))
                {
                    inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize, seqLen, Column });
                    inputEmbs = g.Add(inputEmbs, posEmbeddingPeekViewExp, inPlace: true);
                    inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize * seqLen, Column });
                }
            }

            posEmbeddingPeek.Dispose();

            inputEmbs = g.Dropout(inputEmbs, batchSize, dropoutRatio, inPlace: true);

            return inputEmbs;
        }

        public static WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false, DType elementType = DType.Float32)
        {
            Logger.WriteLine($"Building position weights tensor. Row = '{row}', Column = '{column}', DeviceId = '{deviceId}', Name = '{name}', Trainable = '{isTrainable}'");

            WeightTensor t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable, needGradient: isTrainable, dtype: elementType);
            float[] posWeights = new float[row * column];

            float numTimescales = (float)column / 2;
            float logTimescaleIncrement = (float)(Math.Log(10000.0f) / (numTimescales - 1.0f));

            for (int p = 0; p < row; ++p)
            {
                for (int i = 0; i < numTimescales; i++)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));

                    posWeights[p * column + i] = (float)Math.Sin(v);
                    posWeights[p * column + (int)numTimescales + i] = (float)Math.Cos(v);
                }
            }

            if (elementType == DType.Float16)
            {
                Tensor tmp = new Tensor(t.Allocator, DType.Float32, t.Sizes);
                tmp.CopyFrom(posWeights);
                Ops.Float2Half(t.TWeight, tmp);
                tmp.Dispose();
            }
            else
            {
                t.TWeight.CopyFrom(posWeights);
            }

            return t;
        }
    }
}