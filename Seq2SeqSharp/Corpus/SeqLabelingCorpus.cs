using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public class SeqLabelingCorpus : ParallelCorpus<SeqLabelingCorpusBatch>
    {
        private (string, string) ConvertSequenceLabelingFormatToParallel(string filePath)
        {
            List<string> srcLines = new List<string>();
            List<string> tgtLines = new List<string>();

            List<string> currSrcLine = new List<string>();
            List<string> currTgtLine = new List<string>();
            foreach (var line in File.ReadAllLines(filePath))
            {
                if (String.IsNullOrEmpty(line) == true)
                {
                    //This is a new record

                    srcLines.Add(String.Join(" ", currSrcLine));
                    tgtLines.Add(String.Join(" ", currTgtLine));

                    currSrcLine = new List<string>();
                    currTgtLine = new List<string>();
                }
                else
                {
                    string[] items = line.Split(new char[] { ' ', '\t' });
                    string srcItem = items[0];
                    string tgtItem = items[1];

                    currSrcLine.Add(srcItem);
                    currTgtLine.Add(tgtItem);
                }
            }

            srcLines.Add(String.Join(" ", currSrcLine));
            tgtLines.Add(String.Join(" ", currTgtLine));

            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            File.WriteAllLines(srcFilePath, srcLines);
            File.WriteAllLines(tgtFilePath, tgtLines);

            Logger.WriteLine($"Convert sequence labeling corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }


        public SeqLabelingCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSentLength;
            m_maxTgtSentLength = maxSentLength;
            m_shuffleEnums = shuffleEnums;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();


            (string srcFilePath, string tgtFilePath) = ConvertSequenceLabelingFormatToParallel(corpusFilePath);

            m_srcFileList.Add(srcFilePath);
            m_tgtFileList.Add(tgtFilePath);
        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab) BuildVocabs(int vocabSize = 45000, bool sharedVocab = false)
        {
            Vocab srcVocab = new Vocab();
            Vocab tgtVocab = new Vocab();

            Logger.WriteLine($"Building vocabulary from corpus.");

            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            Dictionary<string, int> t_d = new Dictionary<string, int>();

            foreach (SeqLabelingCorpusBatch sntPairBatch in this)
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
                    for (int i = 0, n = item2.Length; i < n; i++)
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

            return (srcVocab, tgtVocab);
        }
    }
}
