using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp
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
        public ConcurrentDictionary<string, int> SrcWordToIndex;
        public ConcurrentDictionary<string, int> TgtWordToIndex;

        private ConcurrentDictionary<int, string> m_srcIndexToWord;
        private List<string> m_srcVocab = new List<string>();
        private ConcurrentDictionary<int, string> m_tgtIndexToWord;
        private List<string> m_tgtVocab = new List<string>();


        public int SourceWordSize => m_srcIndexToWord.Count;
        public int TargetWordSize => m_tgtIndexToWord.Count;

        public List<string> SrcVocab => m_srcVocab.GetRange(3, m_srcVocab.Count - 3);

        public List<string> TgtVocab => m_tgtVocab.GetRange(3, m_tgtVocab.Count - 3);

        public Vocab()
        {
            CreateIndex();
        }

        private void CreateIndex()
        {
            SrcWordToIndex = new ConcurrentDictionary<string, int>();
            m_srcIndexToWord = new ConcurrentDictionary<int, string>();
            m_srcVocab = new List<string>();

            TgtWordToIndex = new ConcurrentDictionary<string, int>();
            m_tgtIndexToWord = new ConcurrentDictionary<int, string>();
            m_tgtVocab = new List<string>();

            m_srcVocab.Add(ParallelCorpus.EOS);
            m_srcVocab.Add(ParallelCorpus.BOS);
            m_srcVocab.Add(ParallelCorpus.UNK);

            SrcWordToIndex[ParallelCorpus.EOS] = (int)SENTTAGS.END;
            SrcWordToIndex[ParallelCorpus.BOS] = (int)SENTTAGS.START;
            SrcWordToIndex[ParallelCorpus.UNK] = (int)SENTTAGS.UNK;

            m_srcIndexToWord[(int)SENTTAGS.END] = ParallelCorpus.EOS;
            m_srcIndexToWord[(int)SENTTAGS.START] = ParallelCorpus.BOS;
            m_srcIndexToWord[(int)SENTTAGS.UNK] = ParallelCorpus.UNK;

            m_tgtVocab.Add(ParallelCorpus.EOS);
            m_tgtVocab.Add(ParallelCorpus.BOS);
            m_tgtVocab.Add(ParallelCorpus.UNK);

            TgtWordToIndex[ParallelCorpus.EOS] = (int)SENTTAGS.END;
            TgtWordToIndex[ParallelCorpus.BOS] = (int)SENTTAGS.START;
            TgtWordToIndex[ParallelCorpus.UNK] = (int)SENTTAGS.UNK;

            m_tgtIndexToWord[(int)SENTTAGS.END] = ParallelCorpus.EOS;
            m_tgtIndexToWord[(int)SENTTAGS.START] = ParallelCorpus.BOS;
            m_tgtIndexToWord[(int)SENTTAGS.UNK] = ParallelCorpus.UNK;
        }

        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        /// <param name="srcVocabFilePath"></param>
        /// <param name="tgtVocabFilePath"></param>
        public Vocab(string srcVocabFilePath, string tgtVocabFilePath)
        {
            Logger.WriteLine("Loading vocabulary files...");
            string[] srcVocab = File.ReadAllLines(srcVocabFilePath);
            string[] tgtVocab = File.ReadAllLines(tgtVocabFilePath);

            CreateIndex();

            //Build word index for both source and target sides
            int q = 3;
            foreach (string line in srcVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                if (ParallelCorpus.IsPreDefinedToken(word) == false)
                {
                    m_srcVocab.Add(word);
                    SrcWordToIndex[word] = q;
                    m_srcIndexToWord[q] = word;
                    q++;
                }
            }

            q = 3;
            foreach (string line in tgtVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                if (ParallelCorpus.IsPreDefinedToken(word) == false)
                {
                    m_tgtVocab.Add(word);
                    TgtWordToIndex[word] = q;
                    m_tgtIndexToWord[q] = word;
                    q++;
                }
            }

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="trainCorpus"></param>
        /// <param name="minFreq"></param>
        public Vocab(ParallelCorpus trainCorpus, int minFreq = 1)
        {
            Logger.WriteLine($"Building vocabulary from given training corpus.");
            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            Dictionary<string, int> t_d = new Dictionary<string, int>();

            CreateIndex();

            foreach (SntPairBatch sntPairBatch in trainCorpus)
            {
                foreach (SntPair sntPair in sntPairBatch.SntPairs)
                {
                    string[] item = sntPair.SrcSnt;
                    for (int i = 0, n = item.Length; i < n; i++)
                    {
                        string txti = item[i];
                        if (s_d.Keys.Contains(txti)) { s_d[txti] += 1; }
                        else { s_d.Add(txti, 1); }
                    }

                    string[] item2 = sntPair.TgtSnt;
                    for (int i = 0, n = item2.Length; i < n; i++)
                    {
                        string txti = item2[i];
                        if (t_d.Keys.Contains(txti)) { t_d[txti] += 1; }
                        else { t_d.Add(txti, 1); }
                    }
                }
            }


            int q = 3;
            foreach (KeyValuePair<string, int> ch in s_d)
            {
                if (ch.Value >= minFreq && ParallelCorpus.IsPreDefinedToken(ch.Key) == false)
                {
                    // add word to vocab
                    SrcWordToIndex[ch.Key] = q;
                    m_srcIndexToWord[q] = ch.Key;
                    m_srcVocab.Add(ch.Key);
                    q++;
                }

            }
            Logger.WriteLine($"Source language Max term id = '{q}'");


            q = 3;
            foreach (KeyValuePair<string, int> ch in t_d)
            {
                if (ch.Value >= minFreq && ParallelCorpus.IsPreDefinedToken(ch.Key) == false)
                {
                    // add word to vocab
                    TgtWordToIndex[ch.Key] = q;
                    m_tgtIndexToWord[q] = ch.Key;
                    m_tgtVocab.Add(ch.Key);
                    q++;
                }

            }

            Logger.WriteLine($"Target language Max term id = '{q}'");
        }


        public void DumpTargetVocab(string fileName)
        {
            List<string> lines = new List<string>();
            foreach (KeyValuePair<int, string> pair in m_tgtIndexToWord)
            {
                lines.Add($"{pair.Value}\t{pair.Key}");
            }

            File.WriteAllLines(fileName, lines);
        }


        public void DumpSourceVocab(string fileName)
        {
            List<string> lines = new List<string>();
            foreach (KeyValuePair<int, string> pair in m_srcIndexToWord)
            {
                lines.Add($"{pair.Value}\t{pair.Key}");
            }

            File.WriteAllLines(fileName, lines);
        }


        public List<string> ConvertTargetIdsToString(List<int> idxs)
        {
            List<string> result = new List<string>();
            foreach (int idx in idxs)
            {
                string letter = ParallelCorpus.UNK;
                if (m_tgtIndexToWord.ContainsKey(idx))
                {
                    letter = m_tgtIndexToWord[idx];
                }
                result.Add(letter);
            }

            return result;
        }

        public int GetSourceWordIndex(string word, bool logUnk = false)
        {
            if (!SrcWordToIndex.TryGetValue(word, out int id))
            {
                id = (int)SENTTAGS.UNK;
                if (logUnk)
                {
                    Logger.WriteLine($"Source word '{word}' is UNK");
                }
            }
            return id;
        }

        public int GetTargetWordIndex(string word)
        {
            if (!TgtWordToIndex.TryGetValue(word, out int id))
            {
                id = (int)SENTTAGS.UNK;
            }
            return id;
        }
    }
}
