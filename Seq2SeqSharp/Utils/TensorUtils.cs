using AdvUtils;
using Seq2SeqSharp.Corpus;
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
        public static IWeightTensor CreateTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, 
            IWeightTensor segmentEmbedding, Vocab vocab, float scaleFactor = 1.0f, bool enableTagEmbedding = false)
        {
            int batchSize = seqs.Count;
            int seqLen = seqs[0].Count;

            float[] idxs = new float[batchSize * seqLen];
            float[] segIdxs = new float[batchSize * seqLen];
            float[] tagIdxs = new float[batchSize * seqLen];

            for (int i = 0; i < batchSize; i++)
            {
                int segIdx = 0;
                int currTagIdx = -1;
                string currTagName = String.Empty;
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


                    if (enableTagEmbedding)
                    {
                        if (token.StartsWith("<") && token.EndsWith(">") && BuildInTokens.IsPreDefinedToken(token) == false)
                        {
                            if (token[1] == '/')
                            {
                                string closedTagName = token.Substring(2, token.Length - 3);
                                if (closedTagName != currTagName)
                                {
                                    throw new DataMisalignedException($"Tag '{currTagName}' and '{closedTagName}' are not paired.");
                                }
                                currTagIdx = -1;
                                currTagName = String.Empty;
                            }
                            else
                            {
                                if (currTagIdx != -1)
                                {
                                    throw new DataMisalignedException($"Tag '{currTagName}' is still opening, you must close it before opening another tag.");
                                }

                                currTagIdx = seqs[i][j];
                                currTagName = token.Substring(1, token.Length - 2);
                            }
                        }
                        else
                        {
                            tagIdxs[i * seqLen + j] = currTagIdx;
                        }
                    }
                }

                if (currTagIdx != -1)
                {
                    throw new DataMisalignedException($"Tag '{currTagName}' is still opening at the end of the sentence.");
                }
            }

            IWeightTensor tagEmbeddings = null;
            if (enableTagEmbedding)
            {
                tagEmbeddings = g.IndexSelect(embeddingsTensor, tagIdxs, clearWeights: true);
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

            if (tagEmbeddings != null)
            {
                embeddingRst = g.Add(embeddingRst, tagEmbeddings);
            }

            return embeddingRst;

        }
    }
}
