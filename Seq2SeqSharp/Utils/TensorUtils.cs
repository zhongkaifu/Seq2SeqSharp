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

            Dictionary<string, List<int>> tag2Offsets = new Dictionary<string, List<int>>();

            for (int i = 0; i < batchSize; i++)
            {
                int segIdx = 0;
                HashSet<string> currTags = new HashSet<string>(); //keep all opening tags
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
                                string tag = token.Substring(2, token.Length - 3);
                                currTags.Remove(tag);

                                //Logger.WriteLine($"Closed tag: '{tag}'");
                            }
                            else
                            {
                                string tag = token.Substring(1, token.Length - 2);
                                currTags.Add(tag);

                                //Logger.WriteLine($"Openning tag: '{tag}'");
                            }
                        }
                        else
                        {
                            foreach (var tag in currTags)
                            {
                                if (tag2Offsets.ContainsKey(tag) == false)
                                {
                                    tag2Offsets.Add(tag, new List<int>());
                                }
                                tag2Offsets[tag].Add(i * seqLen + j);

                                //Logger.WriteLine($"Adding tag: '{tag}' to seq '{i}' offset '{j}'");
                            }
                        }
                    }
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

            if (enableTagEmbedding)
            {
                Dictionary<IWeightTensor, List<int>> tensor2offsets = new Dictionary<IWeightTensor, List<int>>();
                foreach (var pair in tag2Offsets)
                {
                    string tagName = pair.Key;
                    int tagId = vocab.GetWordIndex(tagName);
                    IWeightTensor tagTensor = g.Peek(embeddingsTensor, 0, tagId);
                    tensor2offsets.Add(tagTensor, pair.Value);
                }

                if (tensor2offsets.Count > 0)
                {
                    var tagEmbeddings = g.IndexCopy(embeddingRst.Sizes, tensor2offsets);
                    embeddingRst = g.Add(embeddingRst, tagEmbeddings);
                }
            }

            return embeddingRst;

        }
    }
}
