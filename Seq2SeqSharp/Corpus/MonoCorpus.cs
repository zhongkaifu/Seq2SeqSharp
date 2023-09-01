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
    public class MonoCorpus<T> : ICorpus<T> where T : ISntPairBatch, new()
    {
        internal int m_maxTgtTokenSize = 32;
        internal int m_maxTokenSizePerBatch = 1;
        internal List<string> m_tgtFileList;
        internal ShuffleEnums m_shuffleEnums;

        private bool m_showTokenDist = true;

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        public string CorpusName { get; set; }

        private TooLongSequence m_tooLongSequence = TooLongSequence.Ignore;

        private string m_indexedDataSetFilePath = "";
        private int m_batchNumInTotal = 0;

        public List<Dictionary<string, int>> CountTokenFreqs()
        {
            List<Dictionary<string, int>> td = new List<Dictionary<string, int>>();

            for (int i = 0; i < m_tgtFileList.Count; i++)
            {
                Logger.WriteLine($"Start to count token frequency in '{m_tgtFileList[i]}'.");
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    if (srTgt.EndOfStream)
                    {
                        break;
                    }

                    string tgtLine = srTgt.ReadLine();

                    if (tgtLine.IsNullOrEmpty())
                    {
                        break;
                    }

                    string[] tgtGroups = tgtLine.Split('\t');


                    if (td.Count == 0)
                    {
                        for (int j = 0; j < tgtGroups.Length; j++)
                        {
                            td.Add(new Dictionary<string, int>());
                        }
                    }

                    for (int j = 0; j < tgtGroups.Length; j++)
                    {
                        string[] tgtTokens = tgtGroups[j].Split(' ');
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


            for (int j = 0; j < td.Count; j++)
            {
                Logger.WriteLine($"Original token size at group '{j}' target = '{td[j].Count}'");
            }

            return td;
        }


        private (Dictionary<long, LinkedList<long>>, Dictionary<long, long>, string) BuildIndex()
        {
            Logger.WriteLine($"Start to build index for data set.");
            SortedDictionary<int, int> dictTgtLenDist = new SortedDictionary<int, int>();
            int corpusSize = 0;
            int tooLongTgtSntCnt = 0;
            string randomFileName = Path.GetRandomFileName();
            Logger.WriteLine($"Loading and shuffling corpus from '{m_tgtFileList.Count}' files.");

            string binaryDataSetFilePath = randomFileName + ".tmp";
            BinaryWriter bw = new BinaryWriter(new FileStream(binaryDataSetFilePath, FileMode.Create));

            Dictionary<long, LinkedList<long>> len2offsets = new Dictionary<long, LinkedList<long>>();
            Dictionary<long, long> len2lengths = new Dictionary<long, long>();

            for (int i = 0; i < m_tgtFileList.Count; i++)
            {
                StreamReader srTgt = new StreamReader(m_tgtFileList[i]);

                while (true)
                {
                    if (srTgt.EndOfStream)
                    {
                        break;
                    }

                    RawSntPair rawSntPair = new RawSntPair(null, srTgt.ReadLine(), 0, m_maxTgtTokenSize, m_tooLongSequence == TooLongSequence.Truncation);
                    if (rawSntPair.IsEmptyPair())
                    {
                        break;
                    }

                    if (m_showTokenDist)
                    {
                        if (dictTgtLenDist.ContainsKey(rawSntPair.TgtTokenSize / 100) == false)
                        {
                            dictTgtLenDist.Add(rawSntPair.TgtTokenSize / 100, 0);
                        }
                        dictTgtLenDist[rawSntPair.TgtTokenSize / 100]++;
                    }

                    bool hasTooLongSent = false;
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
                    bw.Write(rawSntPair.TgtSnt);

                    long length = 0;
                    if (m_shuffleEnums == ShuffleEnums.NoPadding)
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
                        len2lengths.Add(length, 0);
                    }
                    len2offsets[length].AddLast(offset);
                    len2lengths[length]++;

                    Interlocked.Increment(ref corpusSize);
                }

                srTgt.Close();
            }

            bw.Close();

            Logger.WriteLine($"Shuffled '{corpusSize}' sentence pairs.");

            if (tooLongTgtSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongTgtSntCnt} target sentences are longer than '{m_maxTgtTokenSize}' tokens, ignore them.");
            }

            if (m_showTokenDist)
            {
                Logger.WriteLine($"AggregateLength = '{m_shuffleEnums}'");
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

            return (len2offsets, len2lengths, binaryDataSetFilePath);
        }


        public long GetNextLength(Dictionary<long, LinkedList<long>> len2offsets, Dictionary<long, long> len2counts)
        {
            long totalItems = 0;
            foreach (var pair in len2offsets)
            {
                totalItems += len2counts[pair.Key];
            }

            int rndItems = rnd.Next((int)totalItems);
            totalItems = 0;
            foreach (var pair in len2offsets)
            {
                long length = len2counts[pair.Key];
                if (totalItems <= rndItems && totalItems + length >= rndItems)
                {
                    return pair.Key;
                }
                totalItems += length;
            }

            return -1;
        }

        public void PrepareDataSet()
        {
            try
            {
                m_batchNumInTotal = 0;
                (var length2offsets, var length2counts, string tmpDataSetFilePath) = BuildIndex();
                Logger.WriteLine($"Start to sort and shuffle data set by length.");

                m_indexedDataSetFilePath = tmpDataSetFilePath + ".sorted";
                using (BinaryWriter bw = new BinaryWriter(new FileStream(m_indexedDataSetFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 40960000)))
                using (MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(tmpDataSetFilePath))
                using (MemoryMappedViewStream mms = mmf.CreateViewStream())
                {
                    using (BinaryReader br = new BinaryReader(mms))
                    {
                        while (length2offsets.Count > 0)
                        {
                            long length = GetNextLength(length2offsets, length2counts);
                            LinkedList<long> offsets = length2offsets[length];

                            int totalTgtTokenSize = 0;
                            int sentSize = 0;
                            List<string> tgtLines = new List<string>();
                            while (totalTgtTokenSize < m_maxTokenSizePerBatch && offsets.Any())
                            {
                                long offset = offsets.First.Value;
                                offsets.RemoveFirst();
                                length2counts[length]--;

                                br.BaseStream.Seek(offset, SeekOrigin.Begin);
                                string tgtLine = br.ReadString();
                                totalTgtTokenSize += tgtLine.Split(' ').Length;
                                tgtLines.Add(tgtLine);


                                sentSize++;
                            }

                            bw.Write(sentSize);
                            bw.Write(String.Join("\n", tgtLines));

                            m_batchNumInTotal++;
                            if (m_batchNumInTotal % 10000 == 0)
                            {
                                Logger.WriteLine($"Batch '{m_batchNumInTotal}' has been processed.");
                            }


                            if (offsets.Any() == false)
                            {
                                length2offsets.Remove(length);
                                length2counts.Remove(length);
                            }
                        }

                        bw.Write(-1);
                    }
                }

                File.Delete(tmpDataSetFilePath);

                Logger.WriteLine($"Finished to sort and shuffle data set by length. Total batch size = '{m_batchNumInTotal}'");
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to prepare data set: '{err.Message}', Call Stack = '{err.StackTrace}'");
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            if (String.IsNullOrEmpty(m_indexedDataSetFilePath) || File.Exists(m_indexedDataSetFilePath) == false)
            {
                PrepareDataSet();
            }
            else
            {
                Logger.WriteLine($"Use existing indexed data set '{m_indexedDataSetFilePath}'");
            }

            int batchIdx = 0;
            int currentBatchPercent = 0;

            using (MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(m_indexedDataSetFilePath))
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
                        string[] tgtLines = br.ReadString().Split("\n");
                        batchIdx++;

                        T batch;
                        for (int i = 0; i < sizeInBatch; i++)
                        {
                            var tgtLine = tgtLines[i];

                            if (m_batchNumInTotal > 0)
                            {
                                if ((100 * batchIdx / m_batchNumInTotal) > currentBatchPercent)
                                {
                                    Logger.WriteLine($"Processing batch '{batchIdx}/{m_batchNumInTotal}'."); // The '{i}th' record in this batch is: Target = '{tgtLine}'");
                                    currentBatchPercent++;
                                }
                            }

                            SntPair sntPair = new SntPair(tgtLine, tgtLine);
                            outputs.Add(sntPair);

                            if (outputs.Count == m_maxTokenSizePerBatch)
                            {
                                batch = new T();
                                batch.CreateBatch(outputs);
                                yield return batch;

                                outputs = new List<SntPair>();
                            }
                        }

                        if (outputs.Count > 0)
                        {
                            batch = new T();
                            batch.CreateBatch(outputs);
                            yield return batch;
                        }
                    }
                }
            }

            File.Delete(m_indexedDataSetFilePath);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public MonoCorpus()
        {

        }

        public MonoCorpus(string corpusFilePath, string tgtLangName, int maxTokenSizePerBatch, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore, string indexedFilePath = "")
        {
            Logger.WriteLine($"Loading mono corpus from '{corpusFilePath}' Files search pattern '*.{tgtLangName}.snt' MaxTgtSentLength = '{maxTgtSentLength}', aggregateLengthForShuffle = '{shuffleEnums}', TooLongSequence = '{tooLongSequence}'");
            m_maxTokenSizePerBatch = maxTokenSizePerBatch;
            m_maxTgtTokenSize = maxTgtSentLength;
            m_tooLongSequence = tooLongSequence;
            m_shuffleEnums = shuffleEnums;
            CorpusName = corpusFilePath;
            m_indexedDataSetFilePath = indexedFilePath;

            m_tgtFileList = new List<string>();
            string[] files = Directory.GetFiles(corpusFilePath, $"*.{tgtLangName}.snt", SearchOption.TopDirectoryOnly);
            m_tgtFileList.AddRange(files);           
        }
    }
}
