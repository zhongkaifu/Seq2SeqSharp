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
        public static void Scatter(IWeightTensor res, IWeightTensor source, IWeightTensor indices, int dim)
        {
            WeightTensor i = indices as WeightTensor;
            WeightTensor s = source as WeightTensor;
            WeightTensor r = res as WeightTensor;

            Ops.Scatter(r.TWeight, s.TWeight, dim, i.TWeight);
        }

        public static void ScatterFill(IWeightTensor res, float val, IWeightTensor indices, int dim)
        {
            WeightTensor i = indices as WeightTensor;
            WeightTensor r = res as WeightTensor;

            Ops.ScatterFill(r.TWeight, val, dim, i.TWeight);
        }

        public static IWeightTensor ExtractTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, List<int> seqOriginalLengths, IWeightTensor segmentEmbedding, Vocab vocab)
        {
            int batchSize = seqs.Count;
            int seqLen = seqs[0].Count;

            float[] idxs = new float[batchSize * seqLen];
            float[] segIdxs = new float[batchSize * seqLen];

            for (int i = 0; i < batchSize; i++)
            {
                int segIdx = 0;
                for (int j = 0; j < seqLen; j++)
                {
                    idxs[i * seqLen + j] = seqs[i][j];
                    segIdxs[i * seqLen + j] = segIdx;

                    string token = vocab.GetString(seqs[i][j]);
                    if (token == ParallelCorpus.SEP)
                    {
                        //A new segment
                        segIdx++;
                    }
                }
            }

            if (segmentEmbedding == null)
            {
                return g.IndexSelect(embeddingsTensor, idxs);
            }
            else
            {
                return g.Add(g.IndexSelect(embeddingsTensor, idxs), g.IndexSelect(segmentEmbedding, segIdxs));
            }
        }
    }
}
