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

using AdvUtils;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Corpus
{
    public class CorpusBatch : ISntPairBatch
    {
        public List<List<string>> SrcBatchTokens = null; // shape [batch_size, seq_size]
        public List<List<string>> TgtBatchTokens = null; // shape [batch_size, seq_size]

        public int BatchSize => SrcBatchTokens.Count;

        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public virtual IPairBatch CloneSrcTokens()
        {
            throw new NotImplementedException();
        }

        public static void TryAddPrefix(List<List<string>> tokens, string prefix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(prefix);
                }
                else
                {
                    if (tokens[i][0] != prefix)
                    {
                        tokens[i].Insert(0, prefix);
                    }
                }
            }
        }


        public static void TryAddSuffix(List<List<string>> tokens, string suffix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(suffix);
                }
                else
                {
                    if (tokens[i][^1] != suffix)
                    {
                        tokens[i].Add(suffix);
                    }
                }
            }
        }

        public virtual void CreateBatch(List<IPair> sntPairs)
        {
            SrcTokenCount = 0;
            TgtTokenCount = 0;

            SrcBatchTokens = new List<List<string>>();
            TgtBatchTokens = new List<List<string>>();

            for (int i = 0; i < sntPairs.Count; i++)
            {
                SntPair pair = sntPairs[i] as SntPair;

                SrcBatchTokens.Add(pair.SrcTokens);
                SrcTokenCount += pair.SrcTokens.Count;

                TgtBatchTokens.Add(pair.TgtTokens);
                TgtTokenCount += pair.TgtTokens.Count;
            }
        }

        public IPairBatch GetRange(int idx, int count)
        {
            CorpusBatch cb = new CorpusBatch();
            cb.SrcBatchTokens = SrcBatchTokens.GetRange(idx, count);
            cb.TgtBatchTokens = TgtBatchTokens.GetRange(idx, count);

            return cb;
        }

        public List<List<string>> GetSrcTokens()
        {
            return SrcBatchTokens;
        }

        public List<List<string>> GetTgtTokens()
        {
            return TgtBatchTokens;
        }

        public List<List<string>> InitializeHypTokens(string prefix)
        {
            List<List<string>> hypTkns = new List<List<string>>();
            for (int i = 0; i < BatchSize; i++)
            {
                if (!prefix.IsNullOrEmpty() )
                {
                    hypTkns.Add(new List<string>() { prefix });
                }
                else
                {
                    hypTkns.Add(new List<string>());
                }
            }

            return hypTkns;
        }



        // count up all words
        public static List<Dictionary<string, long>> s_ds = new List<Dictionary<string, long>>();
        public static List<Dictionary<string, long>> t_ds = new List<Dictionary<string, long>>();



        static public void MergeTokensCountSrcTgt(int srcGroupIdx, int tgtGroupIdx)
        {
            Logger.WriteLine(Logger.Level.debug, $"Merge tokens from source group '{srcGroupIdx}' to target group '{tgtGroupIdx}'");

            foreach (var pair in t_ds[tgtGroupIdx])
            {
                if (s_ds[srcGroupIdx].ContainsKey(pair.Key))
                {
                    s_ds[srcGroupIdx][pair.Key] += pair.Value;
                }
                else
                {
                    s_ds[srcGroupIdx].Add(pair.Key, pair.Value);
                }
            }

            t_ds[tgtGroupIdx] = s_ds[srcGroupIdx];

        }

        static public void ReduceSrcTokensToSingleGroup()
        {
            Logger.WriteLine(Logger.Level.debug, $"Reduce source vocabs group from '{s_ds.Count}' to 1");

            Dictionary<string, long> rst = new Dictionary<string, long>();

            foreach (var dict in s_ds)
            {
                foreach (var pair in dict)
                {
                    if (rst.ContainsKey(pair.Key))
                    {
                        rst[pair.Key] += pair.Value;
                    }
                    else
                    {
                        rst.Add(pair.Key, pair.Value);
                    }

                }
            }

            s_ds.Clear();
            s_ds.Add(rst);
        }
       
        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        /// <param name="sharedSrcTgtVocabGroupMapping">The mappings for shared vocabularies between source side and target side. The values in the mappings are group ids. For example: sharedSrcTgtVocabGroupMapping[0] = 1 means the first group in source
        /// side and the second group in target side are shared vocabulary</param>
        static public (List<Vocab>, List<Vocab>) GenerateVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000, int minFreq = 1)
        {
            Logger.WriteLine($"Building vocabulary from corpus.");

            List<Vocab> srcVocabs = null;
            if (srcVocabSize > 0)
            {
                srcVocabs = InnerBuildVocab(srcVocabSize, s_ds, "Source", minFreq);
            }

            List<Vocab> tgtVocabs = null;
            if (tgtVocabSize > 0)
            {
                tgtVocabs = InnerBuildVocab(tgtVocabSize, t_ds, "Target", minFreq);
            }

            s_ds.Clear();
            t_ds.Clear();

            return (srcVocabs, tgtVocabs);
        }

        private static List<Vocab> InnerBuildVocab(int vocabSize, List<Dictionary<string, long>> ds, string tag, int minFreq = 1)
        {
            List<Vocab> vocabs = new List<Vocab>();

            for (int i = 0; i < ds.Count; i++)
            {
                Vocab vocab = new Vocab();
                SortedDictionary<long, List<string>> sd = new SortedDictionary<long, List<string>>();

                var s_d = ds[i];
                foreach (var kv in s_d)
                {
                    if (sd.ContainsKey(kv.Value) == false)
                    {
                        sd.Add(kv.Value, new List<string>());
                    }
                    sd[kv.Value].Add(kv.Key);
                }

                int q = vocab.IndexToWord.Count;
                foreach (var kv in sd.Reverse())
                {
                    if (kv.Key < minFreq)
                    {
                        break;
                    }

                    foreach (var token in kv.Value)
                    {
                        if (BuildInTokens.IsPreDefinedToken(token) == false)
                        {
                            // add word to vocab
                            vocab.WordToIndex[token] = q;
                            vocab.IndexToWord[q] = token;
                            vocab.Items.Add(token);
                            q++;

                            if (q >= vocabSize)
                            {
                                break;
                            }
                        }
                    }

                    if (q >= vocabSize)
                    {
                        break;
                    }
                }

                if (q % 2 != 0)
                {
                    Logger.WriteLine(Logger.Level.debug, $"Added a pad token into vocabulary for alignment.");

                    string pad = "[PAD_0]";
                    vocab.WordToIndex[pad] = q;
                    vocab.IndexToWord[q] = pad;
                    vocab.Items.Add(pad);
                    q++;
                }

                vocabs.Add(vocab);

                Logger.WriteLine(Logger.Level.debug, $"{tag} Vocab Group '{i}': Original vocabulary size = '{s_d.Count}', Truncated vocabulary size = '{q}', Minimum Token Frequency = '{minFreq}'");

            }

            return vocabs;
        }
    
        public int GetSrcGroupSize()
        {
            return SrcBatchTokens.Count;
        }

        public int GetTgtGroupSize()
        {
            return TgtBatchTokens.Count;
        }

        public virtual void CreateBatch(List<List<string>> srcTokensGroups, List<List<string>> tgtTokensGroups)
        {
            throw new NotImplementedException();
        }
    }
}
