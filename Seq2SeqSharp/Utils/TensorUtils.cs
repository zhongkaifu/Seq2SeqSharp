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
        public static IWeightTensor CreateTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, float[] seqOriginalLengths, 
            IWeightTensor segmentEmbedding, IWeightTensor contextEmbeddings, Vocab vocab, bool applyContextEmbeddingsToEntireSequence = true, float scaleFactor = 1.0f)
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

            List<int> segment0Length = new List<int>();

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

                        if (segIdx == 0)
                        {
                            segment0Length.Add(j);
                        }

                        segIdx++;
                    }
                }

                if (segIdx == 0)
                {
                    segment0Length.Add(seqLen);
                }
            }


            IWeightTensor embeddingRst = g.IndexSelect(embeddingsTensor, idxs);
            if (scaleFactor != 1.0f)
            {
                embeddingRst = g.Mul(embeddingRst, scaleFactor, inPlace: true);
            }

            // Apply segment embeddings to the input sequence embeddings
            if (segmentEmbedding != null)
            {
                embeddingRst = g.Add(embeddingRst, g.IndexSelect(segmentEmbedding, segIdxs));
            }

            // Apply contextual feature embeddings to the input sequence embeddings
            if (contextEmbeddings != null)
            {
                int dim = contextEmbeddings.Columns;
                contextEmbeddings = g.View(contextEmbeddings, dims: new long[] { batchSize, 1, dim });
                contextEmbeddings = g.Expand(contextEmbeddings, dims: new long[] { batchSize, seqLen, dim });

                if (applyContextEmbeddingsToEntireSequence == false)
                {
                    //Only apply contexual feature embeddings to the first segment of the input sequence
                    IWeightTensor featureMaskTensor = g.BuildFeatureMask(seqLen, segment0Length, embeddingsTensor.Columns); //shape: (batch_size, seqLen, dim)
                    contextEmbeddings = g.EltMul(contextEmbeddings, featureMaskTensor);
                }

                embeddingRst = g.Add(embeddingRst, contextEmbeddings);

            }

            return embeddingRst;

        }
    }
}
