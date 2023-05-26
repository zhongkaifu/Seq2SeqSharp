// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.CUDA.ContextState;

namespace Seq2SeqSharp.Utils
{
    public class TensorUtils
    {
        public static void InitDevices(ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f, string[] compilerOptions = null, string mklInstructions = "AVX2", bool enableTensorCore = true, CudaMemoryDeviceAllocatorType allocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool, DType elementType = DType.Float32)
        {
            TensorAllocator.InitDevices(archType, ids, memoryUsageRatio = 0.9f, compilerOptions, mklInstructions, enableTensorCore, allocatorType, elementType);
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
        public static IWeightTensor CreateTokensEmbeddings(List<List<int>> seqs, IComputeGraph g, IWeightTensor embeddingsTensor, 
            IWeightTensor segmentEmbedding, Vocab vocab, float scaleFactor = 1.0f, bool enableTagEmbedding = false, bool amp = false)
        {
            int batchSize = seqs.Count;
            int seqLen = seqs[0].Count;

            float[] idxs = new float[batchSize * seqLen];
            float[] segIdxs = new float[batchSize * seqLen];
            List<float[]> tagIdxsList = new List<float[]>();

            //float[] tagIdxs = new float[batchSize * seqLen];

            for (int i = 0; i < batchSize; i++)
            {
                int segIdx = 0;
                List<int> currTagIdxs = new List<int>();
                int currTagLevel = 0;
                
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
                                currTagLevel--;
                                currTagIdxs[currTagLevel] = -1;
                            }
                            else
                            {
                                //A new opening tag
                                while (tagIdxsList.Count <= currTagLevel)
                                {
                                    float[] tagIdxs = new float[batchSize * seqLen];
                                    Array.Fill(tagIdxs, -1.0f);
                                    tagIdxsList.Add(tagIdxs);
                                }

                                while (currTagIdxs.Count <= currTagLevel)
                                {
                                    currTagIdxs.Add(-1);
                                }

                                currTagIdxs[currTagLevel] = seqs[i][j];

                                currTagLevel++;
                            }
                        }
                        else
                        {
                            for (int k = 0; k < currTagLevel; k++)
                            {
                                tagIdxsList[k][i * seqLen + j] = currTagIdxs[k];

                                //Logger.WriteLine($"Add tag embeddings: '{currTagIdxs[k]}'");
                            }
                        }
                    }
                }
            }

            IWeightTensor tagEmbeddings = null;
            if (enableTagEmbedding)
            {
                for (int k = 0; k < tagIdxsList.Count; k++)
                {
                    var indiceTagEmbs = g.CreateTensorWeights(new long[] { tagIdxsList[k].Length, 1 }, tagIdxsList[k]);
                    var tagEmbeddings_k = g.IndexSelect(embeddingsTensor, indiceTagEmbs, clearWeights: true);
                    if (tagEmbeddings == null)
                    {
                        tagEmbeddings = tagEmbeddings_k;
                    }
                    else
                    {
                        tagEmbeddings = g.Add(tagEmbeddings, tagEmbeddings_k);
                    }
                }
            }

            var indiceEmbs = g.CreateTensorWeights(new long[] { idxs.Length, 1 }, idxs);
            IWeightTensor embeddingRst = g.IndexSelect(embeddingsTensor, indiceEmbs);

            if (amp)
            {
                embeddingRst = g.Float2Half(embeddingRst);
            }


            if (scaleFactor != 1.0f)
            {
                embeddingRst = g.Mul(embeddingRst, scaleFactor, inPlace: true);
            }

            // Apply segment embeddings to the input sequence embeddings
            if (segmentEmbedding != null)
            {
                var indiceSeg = g.CreateTensorWeights(new long[] { segIdxs.Length, 1 }, segIdxs);
                embeddingRst = g.Add(embeddingRst, g.IndexSelect(segmentEmbedding, indiceSeg));
            }

            if (tagEmbeddings != null)
            {
                embeddingRst = g.Add(embeddingRst, tagEmbeddings);
            }

            return embeddingRst;

        }
    }
}
