using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Text;

namespace Seq2SeqSharp.Utils
{
    public class PositionEmbedding
    {

        public static IWeightTensor AddPositionEmbedding(IComputeGraph g, IWeightTensor posEmbedding, int batchSize, IWeightTensor inputEmbs, float dropoutRatio)
        {
            var Column = posEmbedding.Columns;
            int seqLen = inputEmbs.Rows / batchSize;

            inputEmbs = g.Mul(inputEmbs, (float)Math.Sqrt(inputEmbs.Columns));

            using (var posEmbeddingPeek = g.Peek(posEmbedding, 0, 0, seqLen, false))
            {
                using (var posEmbeddingPeekView = g.View(posEmbeddingPeek, runGradient: false, dims: new long[] { 1, seqLen, Column }))
                {
                    using (var posEmbeddingPeekViewExp = g.Expand(posEmbeddingPeekView, runGradient: false, dims: new long[] { batchSize, seqLen, Column }))
                    {
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize, seqLen, Column });
                        inputEmbs = g.Add(inputEmbs, posEmbeddingPeekViewExp, runGradient1: true, runGradient2: false, inPlace: true);
                        inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize * seqLen, Column });
                    }
                }
            }

            inputEmbs = g.Dropout(inputEmbs, batchSize, dropoutRatio, inPlace: true);

            return inputEmbs;
        }

        public static WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false)
        {
            WeightTensor t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);
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

            t.TWeight.CopyFrom(posWeights);

            return t;
        }
    }
}
