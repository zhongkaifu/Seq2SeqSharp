using AdvUtils;
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
    public class RawSntPair
    {
        public string SrcSnt;
        public string TgtSnt;

        public int SrcLength = 0;
        public int TgtLength = 0;
        public RawSntPair(string s, string t)
        {
            SrcSnt = s;
            TgtSnt = t;

            SrcLength = CountWhiteSpace(s);
            TgtLength = CountWhiteSpace(t);

        }

        private int CountWhiteSpace(string s)
        {
            if (String.IsNullOrEmpty(s))
            {
                return 0;
            }

            int cnt = 1;
            bool prevIsSpace = false;
            foreach (char ch in s)
            {
                if (ch == ' ' && prevIsSpace == false)
                {
                    cnt++;
                    prevIsSpace = true;
                }
                else
                {
                    prevIsSpace = false;
                }
            }

            return cnt;

        }

        public bool IsEmptyPair()
        {
            return String.IsNullOrEmpty(SrcSnt) && String.IsNullOrEmpty(TgtSnt);
        }
    }

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
        private readonly int m_maxSrcSentLength = 32;
        private readonly int m_maxTgtSentLength = 32;

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

        private bool m_showTokenDist = true;
        private ShuffleEnums m_shuffleEnums;

        public static bool IsPreDefinedToken(string str)
        {
            return str == EOS || str == BOS || str == UNK;
        }

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);


        private void Shuffle(List<RawSntPair> rawSntPairs)
        {
            if (m_shuffleEnums == ShuffleEnums.Random)
            {
                for (int i = 0; i < rawSntPairs.Count; i++)
                {
                    int idx = rnd.Next(0, rawSntPairs.Count);
                    RawSntPair tmp = rawSntPairs[i];
                    rawSntPairs[i] = rawSntPairs[idx];
                    rawSntPairs[idx] = tmp;
                }

                return;
            }


            //Put sentence pair with same source length into the bucket
            Dictionary<int, List<RawSntPair>> dict = new Dictionary<int, List<RawSntPair>>(); //<source sentence length, sentence pair set>
            foreach (RawSntPair item in rawSntPairs)
            {
                int length = m_shuffleEnums == ShuffleEnums.NoPaddingInSrc ? item.SrcLength : item.TgtLength;

                if (dict.ContainsKey(length) == false)
                {
                    dict.Add(length, new List<RawSntPair>());
                }

                dict[length].Add(item);
            }

            //Randomized the order of sentence pairs with same length in source side
            Parallel.ForEach(dict, pair =>
            //foreach (KeyValuePair<int, List<SntPair>> pair in dict)
            {
                Random rnd2 = new Random(DateTime.Now.Millisecond + pair.Key);

                List<RawSntPair> sntPairList = pair.Value;
                for (int i = 0; i < sntPairList.Count; i++)
                {
                    int idx = rnd2.Next(0, sntPairList.Count);
                    RawSntPair tmp = sntPairList[i];
                    sntPairList[i] = sntPairList[idx];
                    sntPairList[idx] = tmp;
                }
            });


            //Split large bucket to smaller buckets
            Dictionary<int, List<RawSntPair>> dictSB = new Dictionary<int, List<RawSntPair>>();

            foreach (var pair in dict)
            {
                if (pair.Value.Count <= m_batchSize)
                {
                    dictSB.Add(pair.Key, pair.Value);
                }
                else
                {
                    int N = pair.Value.Count / m_batchSize;

                    for (int i = 0; i < N; i++)
                    {
                        var pairs = pair.Value.GetRange(i * m_batchSize, m_batchSize);
                        dictSB.Add(pair.Key + 10000 * i, pairs);
                    }

                    if (pair.Value.Count % m_batchSize != 0)
                    {
                        dictSB.Add(pair.Key + 10000 * N, pair.Value.GetRange(m_batchSize * N, pair.Value.Count % m_batchSize));
                    }
                }
            }

            rawSntPairs.Clear();

            int[] keys = dictSB.Keys.ToArray();
            for (int i = 0; i < keys.Length; i++)
            {
                int idx = rnd.Next(0, keys.Length);
                int tmp = keys[i];
                keys[i] = keys[idx];
                keys[idx] = tmp;
            }

            foreach (int key in keys)
            {
                rawSntPairs.AddRange(dictSB[key]);
            }

        }

        private (string, string) ShuffleAll()
        {
            SortedDictionary<int, int> dictSrcLenDist = new SortedDictionary<int, int>();
            SortedDictionary<int, int> dictTgtLenDist = new SortedDictionary<int, int>();

            string srcShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + ".tmp");
            string tgtShuffledFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + ".tmp");

            Logger.WriteLine($"Shuffling corpus for '{m_srcFileList.Count}' files.");

            StreamWriter swSrc = new StreamWriter(srcShuffledFilePath, false);
            StreamWriter swTgt = new StreamWriter(tgtShuffledFilePath, false);

            List<RawSntPair> sntPairs = new List<RawSntPair>();
            CorpusSize = 0;
            int tooLongSrcSntCnt = 0;
            int tooLongTgtSntCnt = 0;

            for (int i = 0; i < m_srcFileList.Count; i++)
            {
                if (m_showTokenDist)
                {
                    Logger.WriteLine($"Process file '{m_srcFileList[i]}' and '{m_tgtFileList[i]}'");
                }

                StreamReader srSrc = new StreamReader(m_srcFileList[i]);
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    if (srSrc.EndOfStream && srTgt.EndOfStream)
                    {
                        break;
                    }

                    RawSntPair rawSntPair = new RawSntPair(srSrc.ReadLine(), srTgt.ReadLine());
                    if (rawSntPair.IsEmptyPair())
                    {
                        break;
                    }

                    if (dictSrcLenDist.ContainsKey(rawSntPair.SrcLength / 100) == false)
                    {
                        dictSrcLenDist.Add(rawSntPair.SrcLength / 100, 0);
                    }
                    dictSrcLenDist[rawSntPair.SrcLength / 100]++;

                    if (dictTgtLenDist.ContainsKey(rawSntPair.TgtLength / 100) == false)
                    {
                        dictTgtLenDist.Add(rawSntPair.TgtLength / 100, 0);
                    }
                    dictTgtLenDist[rawSntPair.TgtLength / 100]++;


                    bool hasTooLongSent = false;
                    if (rawSntPair.SrcLength > m_maxSrcSentLength)
                    {
                        tooLongSrcSntCnt++;
                        hasTooLongSent = true;
                    }

                    if (rawSntPair.TgtLength > m_maxTgtSentLength)
                    {
                        tooLongTgtSntCnt++;
                        hasTooLongSent = true;
                    }

                    if (hasTooLongSent)
                    {
                        continue;
                    }

                    sntPairs.Add(rawSntPair);
                    CorpusSize++;
                    if (m_blockSize > 0 && sntPairs.Count >= m_blockSize)
                    {
                        Shuffle(sntPairs);
                        foreach (RawSntPair item in sntPairs)
                        {
                            swSrc.WriteLine(item.SrcSnt);
                            swTgt.WriteLine(item.TgtSnt);
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
                foreach (RawSntPair item in sntPairs)
                {
                    swSrc.WriteLine(item.SrcSnt);
                    swTgt.WriteLine(item.TgtSnt);
                }

                sntPairs.Clear();
            }


            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Shuffled '{CorpusSize}' sentence pairs to file '{srcShuffledFilePath}' and '{tgtShuffledFilePath}'.");

            if (tooLongSrcSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongSrcSntCnt} source sentences are longer than '{m_maxSrcSentLength}' tokens, ignore them.");
            }

            if (tooLongTgtSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongTgtSntCnt} target sentences are longer than '{m_maxTgtSentLength}' tokens, ignore them.");
            }

            if (m_showTokenDist)
            {
                Logger.WriteLine($"AggregateSrcLength = '{m_shuffleEnums}'");
                Logger.WriteLine($"Src token length distribution");

                int srcTotalNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcTotalNum += pair.Value;
                }

                int srcAccNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcAccNum += pair.Value;

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value} (acc: {(100.0f * (float)srcAccNum / (float)srcTotalNum).ToString("F")}%)");
                }

                Logger.WriteLine($"Tgt token length distribution");

                int tgtTotalNum = 0;
                foreach (var pair in dictTgtLenDist)
                {
                    tgtTotalNum += pair.Value;
                }

                int tgtAccNum = 0;

                foreach (var pair in dictTgtLenDist)
                {
                    tgtAccNum += pair.Value;

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value}  (acc: {(100.0f * (float)tgtAccNum / (float)tgtTotalNum).ToString("F")}%)");
                }

                m_showTokenDist = false;
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
                    int lastTgtSntLen = -1;
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
                        sntPair.SrcSnt = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                        line = srTgt.ReadLine().ToLower().Trim();
                        if (m_addBOSEOS)
                        {
                            line = $"{BOS} {line} {EOS}";
                        }
                        sntPair.TgtSnt = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);


                        if ((lastTgtSntLen > 0 && m_shuffleEnums == ShuffleEnums.NoPaddingInTgt && lastTgtSntLen != sntPair.TgtSnt.Length) ||
                            (lastSrcSntLen > 0 && m_shuffleEnums == ShuffleEnums.NoPaddingInSrc && lastSrcSntLen != sntPair.SrcSnt.Length) ||
                            outputs.Count > maxOutputsSize)
                        {
                            // InnerShuffle(outputs);
                            for (int i = 0; i < outputs.Count; i += m_batchSize)
                            {
                                int size = Math.Min(m_batchSize, outputs.Count - i);
                                yield return new SntPairBatch(outputs.GetRange(i, size));
                            }

                            outputs.Clear();
                        }

                        outputs.Add(sntPair);

                        lastSrcSntLen = sntPair.SrcSnt.Length;
                        lastTgtSntLen = sntPair.TgtSnt.Length;
                    }

                    // InnerShuffle(outputs);
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


        public static List<List<string>> LeftShiftSnts(List<List<string>> input, string lastTokenToPad)
        {
            List<List<string>> r = new List<List<string>>();

            foreach (var seq in input)
            {
                List<string> rseq = new List<string>();

                rseq.AddRange(seq);
                rseq.RemoveAt(0);
                rseq.Add(lastTokenToPad);

                r.Add(rseq);
            }

            return r;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public ParallelCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSrcSentLength = 32, int maxTgtSentLength = 32, bool addBOSEOS = true, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading parallel corpus from '{corpusFilePath}' for source side '{srcLangName}' and target side '{tgtLangName}' MaxSrcSentLength = '{maxSrcSentLength}',  MaxTgtSentLength = '{maxTgtSentLength}', addBOSEOS = '{addBOSEOS}', aggregateSrcLengthForShuffle = '{shuffleEnums}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSrcSentLength;
            m_maxTgtSentLength = maxTgtSentLength;

            m_addBOSEOS = addBOSEOS;
            m_shuffleEnums = shuffleEnums;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();
            string[] files = Directory.GetFiles(corpusFilePath, $"*.*", SearchOption.TopDirectoryOnly);

            Dictionary<string, string> srcKey2FileName = new Dictionary<string, string>();
            Dictionary<string, string> tgtKey2FileName = new Dictionary<string, string>();

            string srcSuffix = $".{srcLangName}.snt";
            string tgtSuffix = $".{tgtLangName}.snt";

            foreach (string file in files)
            {
                if (file.EndsWith(srcSuffix, StringComparison.InvariantCultureIgnoreCase))
                {
                    string srcKey = file.Substring(0, file.Length - srcSuffix.Length);
                    srcKey2FileName.Add(srcKey, file);

                    Logger.WriteLine($"Add source file '{file}' to key '{srcKey}'");
                }

                if (file.EndsWith(tgtSuffix, StringComparison.InvariantCultureIgnoreCase))
                {
                    string tgtKey = file.Substring(0, file.Length - tgtSuffix.Length);
                    tgtKey2FileName.Add(tgtKey, file);


                    Logger.WriteLine($"Add target file '{file}' to key '{tgtKey}'");
                }
            }

            foreach (var pair in srcKey2FileName)
            {
                m_srcFileList.Add(pair.Value);
                m_tgtFileList.Add(tgtKey2FileName[pair.Key]);
            }

        }
    }
}
