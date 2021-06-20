using AdvUtils;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    /// <summary>
    /// Data format:
    /// Source side: tokens split by space
    /// Target side: [classification tag] \t tokens split by space
    /// </summary>
    public class Seq2SeqClassificationCorpus : ParallelCorpus<Seq2SeqClassificationCorpusBatch>
    {

        public Seq2SeqClassificationCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
            : base(corpusFilePath, srcLangName, tgtLangName, batchSize, shuffleBlockSize, maxSrcSentLength, maxTgtSentLength, shuffleEnums: shuffleEnums)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// For return vocabs: (source vocab, target vocab, classification vocab)
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab, Vocab) BuildVocabs(int vocabSize = 45000, bool sharedVocab = false)
        {
            Vocab srcVocab = new Vocab();
            Vocab tgtVocab = new Vocab();
            Vocab clsVocab = new Vocab();

            Logger.WriteLine($"Building vocabulary from corpus.");

            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            Dictionary<string, int> t_d = new Dictionary<string, int>();
            HashSet<string> setClsTags = new HashSet<string>();

            foreach (Seq2SeqClassificationCorpusBatch sntPairBatch in this)
            {
                foreach (SntPair sntPair in sntPairBatch.SntPairs)
                {
                    string[] item = sntPair.SrcSnt;
                    for (int i = 0, n = item.Length; i < n; i++)
                    {
                        string txti = item[i];
                        if (s_d.Keys.Contains(txti)) { s_d[txti] += 1; }
                        else { s_d.Add(txti, 1); }

                        if (sharedVocab)
                        {
                            if (t_d.Keys.Contains(txti)) { t_d[txti] += 1; }
                            else { t_d.Add(txti, 1); }
                        }
                    }

                    string[] item2 = sntPair.TgtSnt;
                    setClsTags.Add(item2[0]);

                    for (int i = 1, n = item2.Length; i < n; i++)
                    {
                        string txti = item2[i];
                        if (t_d.Keys.Contains(txti)) { t_d[txti] += 1; }
                        else { t_d.Add(txti, 1); }

                        if (sharedVocab)
                        {
                            if (s_d.Keys.Contains(txti)) { s_d[txti] += 1; }
                            else { s_d.Add(txti, 1); }
                        }
                    }
                }
            }

            SortedDictionary<int, List<string>> s_sd = new SortedDictionary<int, List<string>>();
            SortedDictionary<int, List<string>> t_sd = new SortedDictionary<int, List<string>>();

            foreach (var kv in s_d)
            {
                if (s_sd.ContainsKey(kv.Value) == false)
                {
                    s_sd.Add(kv.Value, new List<string>());
                }
                s_sd[kv.Value].Add(kv.Key);
            }

            foreach (var kv in t_d)
            {
                if (t_sd.ContainsKey(kv.Value) == false)
                {
                    t_sd.Add(kv.Value, new List<string>());
                }
                t_sd[kv.Value].Add(kv.Key);
            }


            int q = 3;
            foreach (var kv in s_sd.Reverse())
            {
                foreach (var token in kv.Value)
                {
                    if (BuildInTokens.IsPreDefinedToken(token) == false)
                    {
                        // add word to vocab
                        srcVocab.WordToIndex[token] = q;
                        srcVocab.IndexToWord[q] = token;
                        srcVocab.Items.Add(token);
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

            Logger.WriteLine($"Original source vocabulary size = '{s_d.Count}', Truncated source vocabulary size = '{q}'");

            q = 3;
            foreach (var kv in t_sd.Reverse())
            {
                foreach (var token in kv.Value)
                {
                    if (BuildInTokens.IsPreDefinedToken(token) == false)
                    {
                        // add word to vocab
                        tgtVocab.WordToIndex[token] = q;
                        tgtVocab.IndexToWord[q] = token;
                        tgtVocab.Items.Add(token);
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

            Logger.WriteLine($"Original target vocabulary size = '{s_d.Count}', Truncated target vocabulary size = '{q}'");

            q = 0;
            foreach (var item in setClsTags)
            {
                clsVocab.WordToIndex[item] = q;
                clsVocab.IndexToWord[q] = item;
                clsVocab.Items.Add(item);
                q++;
            }

            Logger.WriteLine($"Classification vocabulary size = '{q}'");

            return (srcVocab, tgtVocab, clsVocab);
        }
    }
}
