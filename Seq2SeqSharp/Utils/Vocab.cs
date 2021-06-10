using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp.Utils
{
    public enum SENTTAGS
    {
        END = 0,
        START,
        UNK
    }

    [Serializable]
    public class Vocab
    {
        public Dictionary<string, int> WordToIndex;
        public Dictionary<int, string> IndexToWord;
        public List<string> Items = new List<string>();
        public int Count => IndexToWord.Count;

        private object locker = new object();

        public Vocab()
        {
            CreateIndex();
        }

        private void CreateIndex()
        {
            WordToIndex = new Dictionary<string, int>();
            IndexToWord = new Dictionary<int, string>();
            Items = new List<string>();

            Items.Add(ParallelCorpus.EOS);
            Items.Add(ParallelCorpus.BOS);
            Items.Add(ParallelCorpus.UNK);

            WordToIndex[ParallelCorpus.EOS] = (int)SENTTAGS.END;
            WordToIndex[ParallelCorpus.BOS] = (int)SENTTAGS.START;
            WordToIndex[ParallelCorpus.UNK] = (int)SENTTAGS.UNK;

            IndexToWord[(int)SENTTAGS.END] = ParallelCorpus.EOS;
            IndexToWord[(int)SENTTAGS.START] = ParallelCorpus.BOS;
            IndexToWord[(int)SENTTAGS.UNK] = ParallelCorpus.UNK;
        }

        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        /// <param name="vocabFilePath"></param>
        public Vocab(string vocabFilePath)
        {
            Logger.WriteLine("Loading vocabulary files...");
            string[] vocab = File.ReadAllLines(vocabFilePath);

            CreateIndex();

            //Build word index for both source and target sides
            int q = 3;
            foreach (string line in vocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                if (ParallelCorpus.IsPreDefinedToken(word) == false)
                {
                    Items.Add(word);
                    WordToIndex[word] = q;
                    IndexToWord[q] = word;
                    q++;
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
                string letter = ParallelCorpus.UNK;
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
                    string letter = ParallelCorpus.UNK;
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
                            letter = ParallelCorpus.UNK;
                        }
                        r.Add(letter);
                    }

                    result.Add(r);
                }
            }

            return result;
        }

        public List<List<List<string>>> ConvertIdsToString(List<List<List<int>>> beam2seqs)
        {
            List<List<List<string>>> result = new List<List<List<string>>>();
            lock (locker)
            {
                foreach (var seqs in beam2seqs)
                {
                    List<List<string>> b = new List<List<string>>();
                    foreach (var seq in seqs)
                    {
                        List<string> r = new List<string>();
                        foreach (int idx in seq)
                        {
                            if (!IndexToWord.TryGetValue(idx, out string letter))
                            {
                                letter = ParallelCorpus.UNK;
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
