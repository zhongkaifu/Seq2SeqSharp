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

    public class SeqClassificationMultiTasksCorpus : ParallelCorpus<SeqClassificationMultiTasksCorpusBatch>
    {
        
        private (string, string) ConvertSequenceClassificationFormatToParallel(string filePath)
        {
            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            StreamWriter swSrc = new StreamWriter(srcFilePath);
            StreamWriter swTgt = new StreamWriter(tgtFilePath);

            foreach (var line in File.ReadAllLines(filePath))
            {
                // Format: [Category Tag1] \t [Category Tag2] \t ... \t [Category TagN] \t [Sequence Text]
                string[] items = line.Split(new char[] { '\t' });
                string srcItem = items[items.Length - 1];
                string tgtItem = String.Join('\t', items, 0, items.Length - 1).Replace(" ", "_").Replace("\t"," ");

                swSrc.WriteLine(srcItem);
                swTgt.WriteLine(tgtItem);
            }

            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Convert sequence classification corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }

        public SeqClassificationMultiTasksCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSentLength;
            m_maxTgtSentLength = maxSentLength;
            m_shuffleEnums = shuffleEnums;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();


            (string srcFilePath, string tgtFilePath) = ConvertSequenceClassificationFormatToParallel(corpusFilePath);

            m_srcFileList.Add(srcFilePath);
            m_tgtFileList.Add(tgtFilePath);
        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, List<Vocab>) BuildVocabs(int vocabSize = 45000)
        {
            Vocab srcVocab = new Vocab();
            List<Vocab> tgtVocabs = new List<Vocab>();

            Logger.WriteLine($"Building vocabulary from corpus.");

            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            List<int> qs = new List<int>();

            foreach (SeqClassificationMultiTasksCorpusBatch sntPairBatch in this)
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
                        while (tgtVocabs.Count < n)
                        {
                            tgtVocabs.Add(new Vocab());
                            qs.Add(3);
                        }

                        string txti = item2[i];
                        if (BuildInTokens.IsPreDefinedToken(txti) == false && tgtVocabs[i].WordToIndex.ContainsKey(txti) == false)
                        {
                            // add word to vocab
                            tgtVocabs[i].WordToIndex[txti] = qs[i];
                            tgtVocabs[i].IndexToWord[qs[i]] = txti;
                            tgtVocabs[i].Items.Add(txti);
                            qs[i]++;
                        }
                    }
                }
            }

            SortedDictionary<int, List<string>> s_sd = new SortedDictionary<int, List<string>>();

            foreach (var kv in s_d)
            {
                if (s_sd.ContainsKey(kv.Value) == false)
                {
                    s_sd.Add(kv.Value, new List<string>());
                }
                s_sd[kv.Value].Add(kv.Key);
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

            return (srcVocab, tgtVocabs);
        }
    }
}
