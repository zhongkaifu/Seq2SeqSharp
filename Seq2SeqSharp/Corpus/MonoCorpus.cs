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
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{

    public class IndexData
    {
        public Dictionary<long, LinkedList<long>> len2offsets = new Dictionary<long, LinkedList<long>>();
        public Dictionary<long, long> len2lengths = new Dictionary<long, long>();
        public string filePath;
    }

    public class MonoCorpus<T> : ICorpus<T> where T : ISntPairBatch, new()
    {
        internal int m_maxTgtTokenSize = 32;
        internal int m_maxTokenSizePerBatch = 1;
        internal List<string> m_tgtFileList;
        internal PaddingEnums m_paddingEnums;

        private bool m_showTokenDist = true;

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        public string CorpusName { get; set; }

        private TooLongSequence m_tooLongSequence = TooLongSequence.Ignore;

        private string m_indexedDataSetFilePath = "";
        private int m_batchNumInTotal = 0;
        private int m_startBatchId = 0;

        private string m_dataPassword = String.Empty;

        public List<Dictionary<string, long>> CountTokenFreqs()
        {
            List<Dictionary<string, long>> td = new List<Dictionary<string, long>>();

            for (int i = 0; i < m_tgtFileList.Count; i++)
            {
                Logger.WriteLine(Logger.Level.debug, $"Start to count token frequency in '{m_tgtFileList[i]}'.");

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
                            td.Add(new Dictionary<string, long>());
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

#if DEBUG            
                for (int j = 0; j < td.Count; j++)
                {
                    Logger.WriteLine(Logger.Level.debug, $"Original token size at group '{j}' target = '{td[j].Count}'");
                }
#endif

            return td;
        }


        private IndexData[] BuildIndex()
        {
            Logger.WriteLine(Logger.Level.debug, $"Start to build index for data set.");

            SortedDictionary<int, int> dictTgtLenDist = new SortedDictionary<int, int>();
            int corpusSize = 0;
            int tooLongTgtSntCnt = 0;

            Logger.WriteLine($"Loading and shuffling corpus from '{m_tgtFileList.Count}' files.");
            object locker = new object();

            IndexData[] indexDatas = new IndexData[m_tgtFileList.Count];

            Parallel.For(0, m_tgtFileList.Count, i =>
            {
                string randomFileName = m_tgtFileList[i];


                string binaryDataSetFilePath = randomFileName + ".tmp";
                BinaryWriter bw = new BinaryWriter(new FileStream(binaryDataSetFilePath, FileMode.Create));

                Dictionary<long, LinkedList<long>> len2offsets = new Dictionary<long, LinkedList<long>>();
                Dictionary<long, long> len2lengths = new Dictionary<long, long>();


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
                        lock (locker)
                        {
                            if (dictTgtLenDist.ContainsKey(rawSntPair.TgtTokenSize / 100) == false)
                            {
                                dictTgtLenDist.Add(rawSntPair.TgtTokenSize / 100, 0);
                            }
                            dictTgtLenDist[rawSntPair.TgtTokenSize / 100]++;
                        }
                    }

                    if (rawSntPair.TgtTokenSize > m_maxTgtTokenSize)
                    {
                        Interlocked.Increment(ref tooLongTgtSntCnt);
                        continue;
                    }

                    long offset = bw.BaseStream.Position;
                    bw.Write(rawSntPair.TgtSnt);

                    long length = 0;
                    if (m_paddingEnums == PaddingEnums.NoPadding)
                    {
                        length = rawSntPair.GroupLenId;
                    }
                    else
                    {
                        length = 0;
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
                bw.Close();

                indexDatas[i] = new IndexData();
                indexDatas[i].len2lengths= len2lengths;
                indexDatas[i].len2offsets= len2offsets;
                indexDatas[i].filePath = binaryDataSetFilePath;
            });
            Logger.WriteLine(Logger.Level.debug, $"Shuffled '{corpusSize}' sentence pairs.");

            if (tooLongTgtSntCnt > 0)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Found {tooLongTgtSntCnt} target sentences are longer than '{m_maxTgtTokenSize}' tokens, ignore them.");
            }

            if (m_showTokenDist)
            {
                //TODO(Zho): executed even if nothing is printed
                {
                    Logger.WriteLine(Logger.Level.debug, $"AggregateLength = '{m_paddingEnums}'");
                    Logger.WriteLine(Logger.Level.debug, $"Tgt token length distribution");
                }

                int tgtTotalNum = 0;
                foreach (var pair in dictTgtLenDist)
                {
                    tgtTotalNum += pair.Value;
                }

                int tgtAccNum = 0;

                foreach (var pair in dictTgtLenDist)
                {
                    tgtAccNum += pair.Value;

                    Logger.WriteLine(Logger.Level.debug, $"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value}  (acc: {100.0f * (float)tgtAccNum / (float)tgtTotalNum:F}%)");
                }

                m_showTokenDist = false;
            }

            Logger.WriteLine(Logger.Level.debug, $"Finished to build index for data set.");

            return indexDatas;
        }


        public long GetNextLength(Dictionary<long, long> len2counts, long totalRecordsNum)
        {
            long rndItems = rnd.NextInt64(totalRecordsNum);
            long totalItems = 0;
            foreach (var pair in len2counts)
            {
                long length = pair.Value;
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
                var indexDatas = BuildIndex();
                object locker = new object();

                Logger.WriteLine(Logger.Level.debug, $"Start to sort and shuffle data set by length.");

                m_indexedDataSetFilePath = Path.GetRandomFileName() + ".sorted";
                using (BinaryWriter bw = new BinaryWriter(new FileStream(m_indexedDataSetFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 40960000)))
                {

                    Parallel.ForEach(indexDatas, (indexData) =>
                {
                    long totalRecordsNum = 0;
                    var length2offsets = indexData.len2offsets;
                    var length2counts = indexData.len2lengths;
                    foreach (var pair in length2offsets)
                    {
                        totalRecordsNum += length2counts[pair.Key];
                    }

                    var tmpDataSetFilePath = indexData.filePath;
                    using (MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(tmpDataSetFilePath))
                    using (MemoryMappedViewStream mms = mmf.CreateViewStream())
                    {
                        using (BinaryReader br = new BinaryReader(mms))
                        {
                            while (length2offsets.Count > 0)
                            {
                                long length = GetNextLength(length2counts, totalRecordsNum);
                                LinkedList<long> offsets = length2offsets[length];

                                int totalTgtTokenSize = 0;
                                int sentSize = 0;
                                List<string> tgtLines = new List<string>();
                                while (totalTgtTokenSize < m_maxTokenSizePerBatch && offsets.Any())
                                {
                                    long offset = offsets.First.Value;
                                    offsets.RemoveFirst();
                                    length2counts[length]--;
                                    totalRecordsNum--;

                                    br.BaseStream.Seek(offset, SeekOrigin.Begin);
                                    string tgtLine = br.ReadString();
                                    totalTgtTokenSize += tgtLine.Split(' ').Length;
                                    tgtLines.Add(tgtLine);


                                    sentSize++;
                                }

                                lock (locker)
                                {
                                    bw.Write(sentSize);
                                    bw.Write(String.Join("\n", tgtLines));
                                }

                                Interlocked.Increment(ref m_batchNumInTotal);
                                if (m_batchNumInTotal % 10000 == 0)
                                {
                                    Logger.WriteLine(Logger.Level.debug, $"Batch '{m_batchNumInTotal}' has been processed.");
                                }


                                if (offsets.Any() == false)
                                {
                                    length2offsets.Remove(length);
                                    length2counts.Remove(length);
                                }
                            }
                        }
                    }
                    File.Delete(tmpDataSetFilePath);
                });

                    bw.Write(-1);
                }
              
                Logger.WriteLine($"Finished to sort and shuffle data set by length. Total batch size = '{m_batchNumInTotal}'");
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.info, $"Failed to prepare data set: '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
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
                Logger.WriteLine(Logger.Level.debug, $"Use existing indexed data set '{m_indexedDataSetFilePath}'");
            }

            int batchIdx = 0;
            int currentBatchPercent = 0;
            MemoryMappedFile mmf = null;
            MemoryMappedViewStream mms = null;
            ZipDecompressor decompressor = null;
            if (m_indexedDataSetFilePath.ToLower().EndsWith(".zip"))
            {
                Logger.WriteLine($"The data set is a zip archive.");
                decompressor = new ZipDecompressor(m_indexedDataSetFilePath, m_dataPassword);
                mms = decompressor.GetMemoryMappedViewStream();
            }
            else
            {
                mmf = MemoryMappedFile.CreateFromFile(m_indexedDataSetFilePath);
                mms = mmf.CreateViewStream();
            }

            using (BinaryReader br = new BinaryReader(mms))
            {
                while (true)
                {
                    int sizeInBatch = br.ReadInt32();
                    if (sizeInBatch < 0)
                    {
                        break;
                    }

                    List<IPair> outputs = new List<IPair>();
                    string[] tgtLines = br.ReadString().Split("\n");
                    batchIdx++;

                    if (batchIdx < m_startBatchId)
                    {
                        continue;
                    }

                    if (batchIdx % 10000 == 0)
                    {
                        Logger.WriteLine($"Processing batch '{batchIdx}'");
                    }

                    T batch;
                    int currentTokenCountsInBatch = 0;
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

                        IPair sntPair = new SntPair(tgtLine, tgtLine);
                        currentTokenCountsInBatch += sntPair.GetTgtTokenCount();
                        outputs.Add(sntPair);

                        if (currentTokenCountsInBatch >= m_maxTokenSizePerBatch)
                        {
                            batch = new T();
                            batch.CreateBatch(outputs);
                            yield return batch;

                            outputs = new List<IPair>();
                            currentTokenCountsInBatch = 0;
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

            if (mms != null)
            {
                mms.Dispose();
            }
            if (mmf != null)
            {
                mmf.Dispose();
            }

            if (decompressor != null)
            {
                decompressor.Dispose();
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

        public MonoCorpus(string corpusFilePath, string tgtLangName, int maxTokenSizePerBatch, int maxTgtSentLength = 32, PaddingEnums paddingEnums = PaddingEnums.AllowPadding, TooLongSequence tooLongSequence = TooLongSequence.Ignore, string indexedFilePath = "", int startBatchId = 0, string dataPassword = "")
        {
            Logger.WriteLine($"Loading mono corpus from '{corpusFilePath}' Files search pattern '*.{tgtLangName}.snt' MaxTgtSentLength = '{maxTgtSentLength}', Token Padding Type = '{paddingEnums}', TooLongSequence = '{tooLongSequence}', Encrypted data set = '{!String.IsNullOrEmpty(dataPassword)}'");
            m_maxTokenSizePerBatch = maxTokenSizePerBatch;
            m_maxTgtTokenSize = maxTgtSentLength;
            m_tooLongSequence = tooLongSequence;
            m_paddingEnums = paddingEnums;
            CorpusName = corpusFilePath;
            m_indexedDataSetFilePath = indexedFilePath;
            m_startBatchId = startBatchId;
            m_dataPassword = dataPassword;

            m_tgtFileList = new List<string>();
            string[] files = Directory.GetFiles(corpusFilePath, $"*.{tgtLangName}.snt", SearchOption.TopDirectoryOnly);
            m_tgtFileList.AddRange(files);           
        }
    }
}
