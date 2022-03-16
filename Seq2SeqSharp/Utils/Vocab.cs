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
        UNK
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
                BuildInTokens.UNK
            };

            WordToIndex[BuildInTokens.EOS] = (int)SENTTAGS.END;
            WordToIndex[BuildInTokens.BOS] = (int)SENTTAGS.START;
            WordToIndex[BuildInTokens.UNK] = (int)SENTTAGS.UNK;

            IndexToWord[(int)SENTTAGS.END] = BuildInTokens.EOS;
            IndexToWord[(int)SENTTAGS.START] = BuildInTokens.BOS;
            IndexToWord[(int)SENTTAGS.UNK] = BuildInTokens.UNK;


            //Add positional OOV tokens
            int idx = WordToIndex.Count;
            for (int i = 0; i < 100; i++)
            {
                string oovToken = $"[OOV_{i}]";

                Items.Add(oovToken);
                WordToIndex[oovToken] = idx;
                IndexToWord[idx] = oovToken;

                idx++;
            }
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
            int q = WordToIndex.Count;
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
        }

        public void MergeVocab(Vocab srcVocab)
        {
            int maxId = 0;

            foreach (var pair in WordToIndex)
            {
                if (pair.Value > maxId)
                {
                    maxId = pair.Value;
                }

            }

            maxId++;
            foreach (var pair in srcVocab.WordToIndex)
            {
                if (WordToIndex.ContainsKey(pair.Key) == false)
                {
                    WordToIndex.Add(pair.Key, maxId);
                    IndexToWord.Add(maxId, pair.Key);
                    Items.Add(pair.Key);
                    maxId++;
                }
            }

        }

 
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

        public List<List<List<string>>> ExtractTokens(List<List<BeamSearchStatus>> beam2batch2seq, List<List<string>> posOOVLists = null)
        {
            List<List<List<string>>> result = new List<List<List<string>>>();
            lock (locker)
            {
                foreach (var batch2seq in beam2batch2seq)
                {
                    List<List<string>> b = new List<List<string>>();
                    int batchIdx = 0;
                    foreach (var seq in batch2seq)
                    {
                        List<string> r = new List<string>();
                        foreach (int idx in seq.OutputIds)
                        {
                            if (!IndexToWord.TryGetValue(idx, out string letter))
                            {
                                letter = BuildInTokens.UNK;
                            }

                            if (posOOVLists != null)
                            {
                                var posOOVList = posOOVLists[batchIdx];
                                if (letter.StartsWith("[OOV_"))
                                {
                                    int oovIdx = int.Parse(letter.Replace("[OOV_", "").Replace("]", ""));

                                    Logger.WriteLine($"Converted word '{posOOVList[oovIdx]}' back from '{letter}'");

                                    letter = posOOVList[oovIdx];
                                }
                            }

                            r.Add(letter);
                        }

                        b.Add(r);

                        batchIdx++;
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

        public List<List<int>> GetWordIndex(List<List<string>> seqs, List<List<string>> posOOVLists = null)
        {
            List<List<int>> result = new List<List<int>>();
            int batchIdx = 0;

            lock (locker)
            {
                foreach (var seq in seqs)
                {
                    List<int> r = new List<int>();
                    foreach (var word in seq)
                    {
                        if (!WordToIndex.TryGetValue(word, out int id))
                        {
                            //The token is not in the vocabulary, so let's convert it to postional OOV token.
                            if (posOOVLists != null)
                            {
                                var posOOVList = posOOVLists[batchIdx];
                                bool newOOV = true;
                                for (int i = 0; i < posOOVList.Count; i++)
                                {
                                    if (word == posOOVList[i])
                                    {
                                        id = WordToIndex[$"[OOV_{i}]"];
                                        newOOV = false;

                                        Logger.WriteLine($"Converted token '{word}' to '[OOV_{i}]'");

                                        break;
                                    }
                                }

                                if (newOOV)
                                {
                                    string newOOVToken = $"[OOV_{posOOVList.Count}]";
                                    if (WordToIndex.ContainsKey(newOOVToken))
                                    {
                                        posOOVList.Add(word);
                                        id = WordToIndex[newOOVToken];

                                        Logger.WriteLine($"Converted token '{word}' to '{newOOVToken}'");
                                    }
                                    else
                                    {
                                        id = (int)SENTTAGS.UNK;
                                    }
                                }
                                
                            }
                            else
                            {
                                id = (int)SENTTAGS.UNK;
                            }
                        }
                        r.Add(id);
                    }

                    result.Add(r);

                    batchIdx++;
                }
            }

            return result;
        }        
    }
}
