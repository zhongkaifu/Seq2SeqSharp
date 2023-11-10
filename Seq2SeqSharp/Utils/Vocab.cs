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
using System.IO;
using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Models;

namespace Seq2SeqSharp.Utils
{
    /// <summary>
    /// 
    /// </summary>
    public enum SENTTAGS
    {
        END = 0,
        START,
        UNK,
        SEP,
        CLS
    }

    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public class Vocab
    {
        private readonly object locker = new object();

        public Dictionary<string, int> WordToIndex;
        public Dictionary<int, string> IndexToWord;
        private bool _IgnoreCase;
        public List<string> Items = new List<string>();

        public int Count => IndexToWord.Count;
        public bool IgnoreCase => _IgnoreCase;
        public Dictionary<string, int> _GetWordToIndex_() => WordToIndex;
        public Dictionary<int, string> _GetIndexToWord_() => IndexToWord;

        public Vocab() => CreateIndex();
        public Vocab( Vocab_4_ProtoBufSerializer v )
        {
            var wordToIndex = v.IgnoreCase ? new Dictionary<string, int>( v._GetWordToIndex_().Count, StringComparer.InvariantCultureIgnoreCase )
                                           : new Dictionary<string, int>( v._GetWordToIndex_().Count );
            foreach ( var p in v._GetWordToIndex_() )
            {
                wordToIndex[ p.Key ] = p.Value;
            }
            WordToIndex = wordToIndex;
            IndexToWord = v._GetIndexToWord_();
            _IgnoreCase = v.IgnoreCase;
        }


        public List<string> GetAllTokens(bool keepBuildInTokens = true)
        {
            if (keepBuildInTokens)
            {
                return Items;
            }
            else
            {
                List<string> results = new List<string>();
                foreach (var item in Items)
                {
                    if (BuildInTokens.IsPreDefinedToken(item) == false)
                    {
                        results.Add(item);
                    }
                }

                return results;
            }
        }

        private void CreateIndex()
        {
            WordToIndex = new Dictionary<string, int>();
            IndexToWord = new Dictionary<int, string>();
            Items = new List<string>
            {
                BuildInTokens.EOS,
                BuildInTokens.BOS,
                BuildInTokens.UNK,
                BuildInTokens.SEP,
                BuildInTokens.CLS
            };

            WordToIndex[BuildInTokens.EOS] = (int)SENTTAGS.END;
            WordToIndex[BuildInTokens.BOS] = (int)SENTTAGS.START;
            WordToIndex[BuildInTokens.UNK] = (int)SENTTAGS.UNK;
            WordToIndex[BuildInTokens.SEP] = (int)SENTTAGS.SEP;
            WordToIndex[BuildInTokens.CLS] = (int)SENTTAGS.CLS;

            IndexToWord[(int)SENTTAGS.END] = BuildInTokens.EOS;
            IndexToWord[(int)SENTTAGS.START] = BuildInTokens.BOS;
            IndexToWord[(int)SENTTAGS.UNK] = BuildInTokens.UNK;
            IndexToWord[(int)SENTTAGS.SEP] = BuildInTokens.SEP;
            IndexToWord[(int)SENTTAGS.CLS] = BuildInTokens.CLS;

        }

        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        public Vocab(string vocabFilePath)
        {
            Logger.WriteLine("Loading vocabulary files...");
            string[] vocab = File.ReadAllLines(vocabFilePath);

            CreateIndex();

            //Build word index for both source and target sides
            int q = IndexToWord.Count;
            foreach (string line in vocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                if (BuildInTokens.IsPreDefinedToken(word) == false)
                {
                    Items.Add(word);
                    WordToIndex[word] = q;
                    IndexToWord[q] = word;
                    q++;
                }
            }

            if (q % 2 != 0)
            {
                if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                    Logger.WriteLine($"Added a pad token into vocabulary for alignment.");

                string pad = "[PAD_0]";
                Items.Add(pad);
                WordToIndex[pad] = q;
                IndexToWord[q] = pad;
                q++;
            }
        }

        //public void MergeVocab(Vocab srcVocab)
        //{
        //    int maxId = 0;

        //    foreach (var pair in WordToIndex)
        //    {
        //        if (pair.Value > maxId)
        //        {
        //            maxId = pair.Value;
        //        }

        //    }

        //    maxId++;
        //    foreach (var pair in srcVocab.WordToIndex)
        //    {
        //        if (WordToIndex.ContainsKey(pair.Key) == false)
        //        {
        //            WordToIndex.Add(pair.Key, maxId);
        //            IndexToWord.Add(maxId, pair.Key);
        //            Items.Add(pair.Key);
        //            maxId++;
        //        }
        //    }

        //}

 
        public void DumpVocab(string fileName)
        {
            List<string> lines = new List<string>();
            foreach (KeyValuePair<int, string> pair in IndexToWord)
            {
                lines.Add($"{pair.Value}\t{pair.Key}");
            }

            File.WriteAllLines(fileName, lines);
        }

        public string GetString(int idx)
        {
            lock (locker)
            {
                string letter = BuildInTokens.UNK;
                if (IndexToWord.ContainsKey(idx))
                {
                    letter = IndexToWord[idx];
                }

                return letter;
            }
        }

        public List<string> ConvertIdsToString(List<float> idxs)
        {
            lock (locker)
            {
                List<string> result = new List<string>();
                foreach (int idx in idxs)
                {
                    string letter = BuildInTokens.UNK;
                    if (IndexToWord.ContainsKey(idx))
                    {
                        letter = IndexToWord[idx];
                    }
                    result.Add(letter);
                }

                return result;
            }
        }

        public List<string> ConvertIdsToString(List<int> idxs)
        {
            lock (locker)
            {
                List<string> result = new List<string>();
                foreach (int idx in idxs)
                {
                    string letter = BuildInTokens.UNK;
                    if (IndexToWord.ContainsKey(idx))
                    {
                        letter = IndexToWord[idx];
                    }
                    result.Add(letter);
                }

                return result;
            }
        }

        public List<List<string>> ConvertIdsToString(List<List<int>> seqs)
        {
            List<List<string>> result = new List<List<string>>();
            lock (locker)
            {
                foreach (var seq in seqs)
                {
                    List<string> r = new List<string>();
                    foreach (int idx in seq)
                    {
                        if (!IndexToWord.TryGetValue(idx, out string letter))
                        {
                            letter = BuildInTokens.UNK;
                        }
                        r.Add(letter);
                    }

                    result.Add(r);
                }
            }

            return result;
        }

        /// <summary>
        /// Convert beam search result to words list
        /// </summary>
        /// <param name="beam2batch2seq"></param>
        /// <returns>Converted list. Format: [Beam_Search_Size, Batch_Size, Sequence_Length]</returns>
        public List<List<List<string>>> CovertToWords(List<List<BeamSearchStatus>> beam2batch2seq)
        {
            List<List<List<string>>> result = new List<List<List<string>>>();
            lock (locker)
            {
                foreach (var batch2seq in beam2batch2seq)
                {
                    List<List<string>> b = new List<List<string>>();
                    foreach (var seq in batch2seq)
                    {
                        List<string> r = new List<string>();
                        foreach (int idx in seq.OutputIds)
                        {
                            if (!IndexToWord.TryGetValue(idx, out string letter))
                            {
                                letter = BuildInTokens.UNK;
                            }
                            r.Add(letter);
                        }

                        b.Add(r);
                    }
                    result.Add(b);
                }
            }

            return result;
        }


        public int GetWordIndex(string word, bool logUnk = false)
        {
            lock (locker)
            {
                if (!WordToIndex.TryGetValue(word, out int id))
                {
                    id = (int)SENTTAGS.UNK;
                    if (logUnk)
                    {
                        Logger.WriteLine($"Source word '{word}' is UNK");
                    }
                }
                return id;
            }
        }

        public List<List<int>> GetWordIndex(List<List<string>> seqs, bool logUnk = false)
        {
            List<List<int>> result = new List<List<int>>();

            lock (locker)
            {
                foreach (var seq in seqs)
                {
                    List<int> r = new List<int>();
                    foreach (var word in seq)
                    {
                        if (!WordToIndex.TryGetValue(word, out int id))
                        {
                            id = (int)SENTTAGS.UNK;
                            if (logUnk)
                            {
                                Logger.WriteLine($"Source word '{word}' is UNK");
                            }
                        }
                        r.Add(id);
                    }

                    result.Add(r);
                }
            }

            return result;
        }        
    }
}
