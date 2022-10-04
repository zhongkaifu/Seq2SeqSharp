// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Threading;

namespace Seq2SeqSharp.Tools
{
    public enum TooLongSequence
    {
        Ignore,
        Truncation
    }

    public interface IParallelCorpus<out T> : IEnumerable<T>
    {
        string CorpusName { get; set; }
    }

    public class ParallelCorpus<T> : IParallelCorpus<T> where T : ISntPairBatch, new()
    {
        internal int m_maxSrcTokenSize = 32;
        internal int m_maxTgtTokenSize = 32;
        internal int m_maxTokenSizePerBatch = 1;
        internal List<string> m_srcFileList;
        internal List<string> m_tgtFileList;
        internal ShuffleEnums m_shuffleEnums;
      
        private bool m_showTokenDist = true;

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        public string CorpusName { get; set; }

        private TooLongSequence m_tooLongSequence = TooLongSequence.Ignore;

        private string m_binaryDataSetFilePath = "";


        public (List<Dictionary<string, int>>, List<Dictionary<string, int>>) CountTokenFreqs()
        {
            List<Dictionary<string, int>> sd = new List<Dictionary<string, int>>();
            List<Dictionary<string, int>> td = new List<Dictionary<string, int>>();

            for (int i = 0; i < m_srcFileList.Count; i++)
            {
                Logger.WriteLine($"Start to count token frequency in '{m_srcFileList[i]}' and '{m_tgtFileList[i]}'.");
                StreamReader srSrc = new StreamReader(m_srcFileList[i]);
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    if (srSrc.EndOfStream && srTgt.EndOfStream)
                    {
                        break;
                    }

                    string srcLine = srSrc.ReadLine();
                    string tgtLine = srTgt.ReadLine();

                    if (srcLine.IsNullOrEmpty() && tgtLine.IsNullOrEmpty())
                    {
                        break;
                    }

                    string[] srcGroups = srcLine.Split('\t');
                    string[] tgtGroups = tgtLine.Split('\t');

                    if (srcGroups.Length != tgtGroups.Length)
                    {
                        throw new InvalidDataException("Inconsistent group size between source side and target side.");
                    }

                    if (sd.Count == 0)
                    {
                        for (int j = 0; j < srcGroups.Length; j++)
                        {
                            sd.Add(new Dictionary<string, int>());
                            td.Add(new Dictionary<string, int>());
                        }
                    }

                    for (int j = 0; j < srcGroups.Length; j++)
                    {
                        string[] srcTokens = srcGroups[j].Split(' ');
                        string[] tgtTokens = tgtGroups[j].Split(' ');


                        foreach (var srcToken in srcTokens)
                        {
                            if (sd[j].ContainsKey(srcToken) == false)
                            {
                                sd[j].Add(srcToken, 0);
                            }
                            sd[j][srcToken]++;
                        }

                        foreach (var tgtToken in tgtTokens)
                        {
                            if (td[j].ContainsKey(tgtToken) == false)
                            {
                                td[j].Add(tgtToken, 0);
                            }
                            td[j][tgtToken]++;
                        }

                    }
                }
            }


            for (int j = 0; j < sd.Count; j++)
            {
                Logger.WriteLine($"Original token size at group '{j}' source = '{sd[j].Count}' target = '{td[j].Count}'");
            }

            return (sd, td);
        }


        private (Dictionary<long, LinkedList<long>>, string) BuildIndex()
        {
            Logger.WriteLine($"Start to build index for data set.");
            SortedDictionary<int, int> dictSrcLenDist = new SortedDictionary<int, int>();
            SortedDictionary<int, int> dictTgtLenDist = new SortedDictionary<int, int>();
            int corpusSize = 0;
            int tooLongSrcSntCnt = 0;
            int tooLongTgtSntCnt = 0;
            string randomFileName = Path.GetRandomFileName();
            Logger.WriteLine($"Loading and shuffling corpus from '{m_srcFileList.Count}' files.");

            string binaryDataSetFilePath = randomFileName + ".tmp";
            BinaryWriter bw = new BinaryWriter(new FileStream(binaryDataSetFilePath, FileMode.Create)); 

            Dictionary<long, LinkedList<long>> len2offsets = new Dictionary<long, LinkedList<long>>();

            for (int i = 0; i < m_srcFileList.Count; i++)
            {
                StreamReader srSrc = new StreamReader(m_srcFileList[i]);
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    if (srSrc.EndOfStream && srTgt.EndOfStream)
                    {
                        break;
                    }

                    RawSntPair rawSntPair = new RawSntPair(srSrc.ReadLine(), srTgt.ReadLine(), m_maxSrcTokenSize, m_maxTgtTokenSize, m_tooLongSequence == TooLongSequence.Truncation);
                    if (rawSntPair.IsEmptyPair())
                    {
                        break;
                    }

                    if (m_showTokenDist)
                    {
                        if (dictSrcLenDist.ContainsKey(rawSntPair.SrcTokenSize / 100) == false)
                        {
                            dictSrcLenDist.Add(rawSntPair.SrcTokenSize / 100, 0);
                        }
                        dictSrcLenDist[rawSntPair.SrcTokenSize / 100]++;

                        if (dictTgtLenDist.ContainsKey(rawSntPair.TgtTokenSize / 100) == false)
                        {
                            dictTgtLenDist.Add(rawSntPair.TgtTokenSize / 100, 0);
                        }
                        dictTgtLenDist[rawSntPair.TgtTokenSize / 100]++;
                    }

                    bool hasTooLongSent = false;
                    if (rawSntPair.SrcTokenSize > m_maxSrcTokenSize)
                    {
                        Interlocked.Increment(ref tooLongSrcSntCnt);
                        hasTooLongSent = true;
                    }

                    if (rawSntPair.TgtTokenSize > m_maxTgtTokenSize)
                    {
                        Interlocked.Increment(ref tooLongTgtSntCnt);
                        hasTooLongSent = true;
                    }

                    if (hasTooLongSent)
                    {
                        continue;
                    }

                    long offset = bw.BaseStream.Position;
                    bw.Write(rawSntPair.SrcTokenSize);
                    bw.Write(rawSntPair.TgtTokenSize);
                    bw.Write(rawSntPair.SrcSnt);
                    bw.Write(rawSntPair.TgtSnt);

                    long length = 0;
                    if (m_shuffleEnums == ShuffleEnums.NoPaddingInSrc)
                    {
                        length = rawSntPair.SrcGroupLenId;
                    }
                    else if (m_shuffleEnums == ShuffleEnums.NoPadding)
                    {
                        length = rawSntPair.GroupLenId;
                    }
                    else
                    {
                        length = rawSntPair.TgtGroupLenId;
                    }

                    if (len2offsets.ContainsKey(length) == false)
                    {
                        len2offsets.Add(length, new LinkedList<long>());
                    }
                    len2offsets[length].AddLast(offset);

                    Interlocked.Increment(ref corpusSize);
                }

                srSrc.Close();
                srTgt.Close();
            }

            bw.Close();

            Logger.WriteLine($"Shuffled '{corpusSize}' sentence pairs.");

            if (tooLongSrcSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongSrcSntCnt} source sentences are longer than '{m_maxSrcTokenSize}' tokens, ignore them.");
            }

            if (tooLongTgtSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongTgtSntCnt} target sentences are longer than '{m_maxTgtTokenSize}' tokens, ignore them.");
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

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value} (acc: {100.0f * (float)srcAccNum / (float)srcTotalNum:F}%)");
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

                    Logger.WriteLine($"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value}  (acc: {100.0f * (float)tgtAccNum / (float)tgtTotalNum:F}%)");
                }

                m_showTokenDist = false;
            }

            Logger.WriteLine($"Finished to build index for data set.");

            return (len2offsets, binaryDataSetFilePath);
        }


        public long GetNextLength(Dictionary<long, LinkedList<long>> len2offsets)
        {
            int totalItems = 0;
            foreach (var pair in len2offsets)
            {
                totalItems += pair.Value.Count;
            }

            int rndItems = rnd.Next(totalItems);
            totalItems = 0;
            foreach (var pair in len2offsets)
            {
                if (totalItems <= rndItems && totalItems + pair.Value.Count >= rndItems)
                {
                    return pair.Key;
                }
                totalItems += pair.Value.Count;
            }

            return -1;
        }

        public void PrepareDataSet()
        {
            (var length2offsets, string tmpDataSetFilePath) = BuildIndex();

            int batchNum = 0;
            Logger.WriteLine($"Start to sort and shuffle data set by length.");

            m_binaryDataSetFilePath = tmpDataSetFilePath + ".sorted";
            using (BinaryWriter bw = new BinaryWriter(new FileStream(m_binaryDataSetFilePath, FileMode.Create)))
            using (BinaryReader br = new BinaryReader(new FileStream(tmpDataSetFilePath, FileMode.Open)))
            {
                while (length2offsets.Count > 0)
                {
                    long length = GetNextLength(length2offsets);
                    LinkedList<long> offsets = length2offsets[length];

                    int totalSrcTokenSize = 0;
                    int totalTgtTokenSize = 0;
                    int sentSize = 0;
                    List<string> srcLines = new List<string>();
                    List<string> tgtLines = new List<string>();
                    while (totalSrcTokenSize + totalTgtTokenSize < m_maxTokenSizePerBatch && offsets.Any())
                    {
                        long offset = offsets.First.Value;
                        offsets.RemoveFirst();

                        br.BaseStream.Seek(offset, SeekOrigin.Begin);
                        totalSrcTokenSize += br.ReadInt32();
                        totalTgtTokenSize += br.ReadInt32();
                        srcLines.Add(br.ReadString());
                        tgtLines.Add(br.ReadString());

                        sentSize++;
                    }

                    bw.Write(sentSize);
                    for (int i = 0; i < sentSize; i++)
                    {
                        bw.Write(srcLines[i]);
                        bw.Write(tgtLines[i]);
                    }

                    batchNum++;
                    if (batchNum % 10000 == 0)
                    {
                        Logger.WriteLine($"Batch '{batchNum}' has been processed.");
                    }


                    if (offsets.Any() == false)
                    {
                        length2offsets.Remove(length);
                    }
                }

                bw.Write(-1);
            }

            File.Delete(tmpDataSetFilePath);

            Logger.WriteLine($"Finished to sort and shuffle data set by length.");
        }

        public IEnumerator<T> GetEnumerator()
        {
            PrepareDataSet();

            using (MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(m_binaryDataSetFilePath))
            using (MemoryMappedViewStream mms = mmf.CreateViewStream())
            {
                using (BinaryReader br = new BinaryReader(mms))
                {
                    while (true)
                    {
                        int sizeInBatch = br.ReadInt32();
                        if (sizeInBatch < 0)
                        {
                            break;
                        }

                        List<SntPair> outputs = new List<SntPair>();
                        for (int i = 0; i < sizeInBatch; i++)
                        {
                            var srcLine = br.ReadString();
                            var tgtLine = br.ReadString();

                            SntPair sntPair = new SntPair(srcLine, tgtLine);
                            outputs.Add(sntPair);
                        }

                        var batch = new T();
                        batch.CreateBatch(outputs);
                        yield return batch;                        
                    }
                }
            }

            File.Delete(m_binaryDataSetFilePath);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public ParallelCorpus()
        {

        }

        public ParallelCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
        {
            Logger.WriteLine($"Loading parallel corpus from '{corpusFilePath}' for source side '{srcLangName}' and target side '{tgtLangName}' MaxSrcSentLength = '{maxSrcSentLength}',  MaxTgtSentLength = '{maxTgtSentLength}', aggregateSrcLengthForShuffle = '{shuffleEnums}', TooLongSequence = '{tooLongSequence}'");
            m_maxTokenSizePerBatch = maxTokenSizePerBatch;
            m_maxSrcTokenSize = maxSrcSentLength;
            m_maxTgtTokenSize = maxTgtSentLength;

            m_tooLongSequence = tooLongSequence;

            m_shuffleEnums = shuffleEnums;
            CorpusName = corpusFilePath;

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
