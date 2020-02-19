using AdvUtils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Seq2SeqSharp.Tools
{
    public class SntPair
    {
        public string[] SrcSnt;
        public string[] TgtSnt;
    }

    public class SntPairBatch
    {
        public List<SntPair> SntPairs;
        public int BatchSize => SntPairs.Count;

        public SntPairBatch(List<SntPair> sntPairs)
        {
            SntPairs = sntPairs;
        }
    }

    public class ParallelCorpus : IEnumerable<SntPairBatch>
    {
        private readonly int m_maxSentLength = 32;
        private readonly int m_blockSize = 1000000;
        private readonly int m_batchSize = 1;
        private readonly bool m_addBOSEOS = true;
        private readonly List<string> m_srcFileList;
        private readonly List<string> m_tgtFileList;

        public int CorpusSize = 0;

        public int BatchSize => m_batchSize;

        public const string EOS = "<END>";
        public const string BOS = "<START>";
        public const string UNK = "<UNK>";


        public static bool IsPreDefinedToken(string str)
        {
            return str == EOS || str == BOS || str == UNK;
        }

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        private void Shuffle(List<SntPair> sntPairs)
        {
            //Put sentence pair with same source length into the bucket
            Dictionary<int, List<SntPair>> dict = new Dictionary<int, List<SntPair>>(); //<source sentence length, sentence pair set>
            foreach (SntPair item in sntPairs)
            {
                if (dict.ContainsKey(item.SrcSnt.Length) == false)
                {
                    dict.Add(item.SrcSnt.Length, new List<SntPair>());
                }
                dict[item.SrcSnt.Length].Add(item);
            }

            //Randomized the order of sentence pairs with same length in source side
            foreach (KeyValuePair<int, List<SntPair>> pair in dict)
            {
                List<SntPair> sntPairList = pair.Value;
                for (int i = 0; i < sntPairList.Count; i++)
                {
                    int idx = rnd.Next(0, sntPairList.Count);
                    SntPair tmp = sntPairList[i];
                    sntPairList[i] = sntPairList[idx];
                    sntPairList[idx] = tmp;
                }
            }

            SortedDictionary<int, List<SntPair>> sdict = new SortedDictionary<int, List<SntPair>>(); //<The bucket size, sentence pair set>
            foreach (KeyValuePair<int, List<SntPair>> pair in dict)
            {
                if (sdict.ContainsKey(pair.Value.Count) == false)
                {
                    sdict.Add(pair.Value.Count, new List<SntPair>());
                }
                sdict[pair.Value.Count].AddRange(pair.Value);
            }

            sntPairs.Clear();

            int[] keys = sdict.Keys.ToArray();
            for (int i = 0; i < keys.Length; i++)
            {
                int idx = rnd.Next(0, keys.Length);
                int tmp = keys[i];
                keys[i] = keys[idx];
                keys[idx] = tmp;
            }

            foreach (int key in keys)
            {
                sntPairs.AddRange(sdict[key]);
            }

        }

        private (string, string) ShuffleAll()
        {
            string srcShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName());
            string tgtShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName());

            Logger.WriteLine("Shuffling corpus...");

            StreamWriter swSrc = new StreamWriter(srcShuffledFilePath, false);
            StreamWriter swTgt = new StreamWriter(tgtShuffledFilePath, false);

            List<SntPair> sntPairs = new List<SntPair>();
            CorpusSize = 0;
            int tooLongSntCnt = 0;
            for (int i = 0; i < m_srcFileList.Count; i++)
            {
                StreamReader srSrc = new StreamReader(m_srcFileList[i]);
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    string line;
                    SntPair sntPair = new SntPair();
                    if ((line = srSrc.ReadLine()) == null)
                    {
                        break;
                    }

                    sntPair.SrcSnt = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    line = srTgt.ReadLine();
                    sntPair.TgtSnt = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    if (sntPair.SrcSnt.Length >= m_maxSentLength || sntPair.TgtSnt.Length >= m_maxSentLength)
                    {
                        tooLongSntCnt++;
                        continue;
                    }

                    sntPairs.Add(sntPair);
                    CorpusSize++;
                    if (m_blockSize > 0 && sntPairs.Count >= m_blockSize)
                    {
                        Shuffle(sntPairs);
                        foreach (SntPair item in sntPairs)
                        {
                            swSrc.WriteLine(string.Join(" ", item.SrcSnt));
                            swTgt.WriteLine(string.Join(" ", item.TgtSnt));
                        }
                        sntPairs.Clear();
                    }
                }

                srSrc.Close();
                srTgt.Close();
            }

            if (sntPairs.Count > 0)
            {
                Shuffle(sntPairs);
                foreach (SntPair item in sntPairs)
                {
                    swSrc.WriteLine(string.Join(" ", item.SrcSnt));
                    swTgt.WriteLine(string.Join(" ", item.TgtSnt));
                }

                sntPairs.Clear();
            }


            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Shuffled '{CorpusSize}' sentence pairs to file '{srcShuffledFilePath}' and '{tgtShuffledFilePath}'.");

            if (tooLongSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongSntCnt} sentences are longer than '{m_maxSentLength}' tokens, ignore them.");
            }

            return (srcShuffledFilePath, tgtShuffledFilePath);
        }

        public IEnumerator<SntPairBatch> GetEnumerator()
        {
            (string srcShuffledFilePath, string tgtShuffledFilePath) = ShuffleAll();

            using (StreamReader srSrc = new StreamReader(srcShuffledFilePath))
            {
                using (StreamReader srTgt = new StreamReader(tgtShuffledFilePath))
                {
                    int lastSrcSntLen = -1;
                    int maxOutputsSize = m_batchSize * 10000;
                    List<SntPair> outputs = new List<SntPair>();

                    while (true)
                    {
                        string line;
                        SntPair sntPair = new SntPair();
                        if ((line = srSrc.ReadLine()) == null)
                        {
                            break;
                        }

                        line = line.ToLower().Trim();
                        if (m_addBOSEOS)
                        {
                            line = $"{BOS} {line} {EOS}";
                        }
                        sntPair.SrcSnt = line.Split(' ');

                        line = srTgt.ReadLine().ToLower().Trim();
                        if (m_addBOSEOS)
                        {
                            line = $"{line} {EOS}";
                        }
                        sntPair.TgtSnt = line.Split(' ');

                        if ((lastSrcSntLen > 0 && lastSrcSntLen != sntPair.SrcSnt.Length) || outputs.Count > maxOutputsSize)
                        {
                            InnerShuffle(outputs);
                            for (int i = 0; i < outputs.Count; i += m_batchSize)
                            {
                                int size = Math.Min(m_batchSize, outputs.Count - i);
                                yield return new SntPairBatch(outputs.GetRange(i, size));
                            }

                            outputs.Clear();
                        }

                        outputs.Add(sntPair);
                        lastSrcSntLen = sntPair.SrcSnt.Length;
                    }

                    InnerShuffle(outputs);
                    for (int i = 0; i < outputs.Count; i += m_batchSize)
                    {
                        int size = Math.Min(m_batchSize, outputs.Count - i);
                        yield return new SntPairBatch(outputs.GetRange(i, size));
                    }
                }
            }

            File.Delete(srcShuffledFilePath);
            File.Delete(tgtShuffledFilePath);
        }

        public static List<List<string>> ConstructInputTokens(List<string> input, bool addBOSEOS = true)
        {
            List<string> inputSeq = new List<string>();

            if (addBOSEOS)
            {
                inputSeq.Add(ParallelCorpus.BOS);
            }

            if (input != null)
            {
                inputSeq.AddRange(input);
            }

            if (addBOSEOS)
            {
                inputSeq.Add(ParallelCorpus.EOS);
            }

            List<List<string>> inputSeqs = new List<List<string>>() { inputSeq };

            return inputSeqs;
        }

        /// <summary>
        /// Pad given sentences to the same length and return their original length
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static List<int> PadSentences(List<List<string>> s, int maxLen = -1)
        {
            List<int> originalLengths = new List<int>();

            if (maxLen <= 0)
            {
                foreach (List<string> item in s)
                {
                    if (item.Count > maxLen)
                    {
                        maxLen = item.Count;
                    }
                }
            }

            for (int i = 0; i < s.Count; i++)
            {
                int count = s[i].Count;
                originalLengths.Add(count);

                for (int j = 0; j < maxLen - count; j++)
                {
                    s[i].Add(ParallelCorpus.EOS);
                }
            }

            return originalLengths;
        }

        /// <summary>
        /// Shuffle given sentence pairs and return the length of the longgest source sentence
        /// </summary>
        /// <param name="sntPairs"></param>
        /// <returns></returns>
        private int InnerShuffle(List<SntPair> sntPairs)
        {
            int maxSrcLen = 0;
            for (int i = 0; i < sntPairs.Count; i++)
            {
                if (sntPairs[i].SrcSnt.Length > maxSrcLen)
                {
                    maxSrcLen = sntPairs[i].SrcSnt.Length;
                }

                int idx = rnd.Next(0, sntPairs.Count);
                SntPair tmp = sntPairs[i];
                sntPairs[i] = sntPairs[idx];
                sntPairs[idx] = tmp;
            }

            return maxSrcLen;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public ParallelCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 32, bool addBOSEOS = true)
        {
            Logger.WriteLine($"Loading corpus from '{corpusFilePath}' for source side '{srcLangName}' and target side '{tgtLangName}' MaxSentLength = '{maxSentLength}', addBOSEOS = '{addBOSEOS}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSentLength = maxSentLength;
            m_addBOSEOS = addBOSEOS;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();
            string[] srcFiles = Directory.GetFiles(corpusFilePath, $"*.{srcLangName}.snt", SearchOption.TopDirectoryOnly);
            foreach (string srcFile in srcFiles)
            {
                string tgtFile = srcFile.ToLower().Replace($".{srcLangName.ToLower()}.", $".{tgtLangName.ToLower()}.");

                m_srcFileList.Add(srcFile);
                m_tgtFileList.Add(tgtFile);
            }
        }
    }
}
