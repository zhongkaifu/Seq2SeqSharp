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

        /// <summary>
        /// Create input embedding from token embeddings, segment embeddings
        /// </summary>
        /// <param name="seqs"></param>
        /// <param name="g"></param>
        /// <param name="embeddingsTensor"></param>
        /// <param name="seqOriginalLengths"></param>
        /// <param name="segmentEmbedding"></param>
        /// <param name="vocab"></param>
        /// <returns>The embedding tensor. shape: (batchsize * seqLen, embedding_dim) </returns>
        public static IWeightTensor ExtractTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, List<int> seqOriginalLengths, IWeightTensor segmentEmbedding, Vocab vocab)
        {
            if (seqs is null)
            {
                throw new ArgumentNullException(nameof(seqs));
            }

            if (g is null)
            {
                throw new ArgumentNullException(nameof(g));
            }

            if (embeddingsTensor is null)
            {
                throw new ArgumentNullException(nameof(embeddingsTensor));
            }

            if (seqOriginalLengths is null)
            {
                throw new ArgumentNullException(nameof(seqOriginalLengths));
            }

            if (vocab is null)
            {
                throw new ArgumentNullException(nameof(vocab));
            }

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
                    if (token == BuildInTokens.SEP)
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
