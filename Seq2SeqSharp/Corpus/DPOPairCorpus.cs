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
    public class DPOPairCorpus<T> : ICorpus<T> where T : ISntPairBatch, new()
    {
        internal int m_maxSrcTokenSize = 32;
        internal int m_maxTgtTokenSize = 32;
        internal int m_maxTokenSizePerBatch = 1;
        internal List<string> m_srcFileList;
        internal List<string> m_tgtFileList;
        internal PaddingEnums m_paddingEnums;

        private bool m_showTokenDist = true;

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        public string CorpusName { get; set; }

        private TooLongSequence m_tooLongSequence = TooLongSequence.Ignore;

        private string m_sortedIndexedDataSetFilePath = "";
        private int m_batchNumInTotal = 0;
        private int m_startBatchId = 0;
        private string m_dataPassword = String.Empty;

        public (List<Dictionary<string, long>>, List<Dictionary<string, long>>) CountTokenFreqs()
        {
            List<Dictionary<string, long>> sd = new List<Dictionary<string, long>>();
            List<Dictionary<string, long>> td = new List<Dictionary<string, long>>();

            for (int i = 0; i < m_srcFileList.Count; i++)
            {
                Logger.WriteLine(Logger.Level.debug, $"Start to count token frequency in '{m_srcFileList[i]}' and '{m_tgtFileList[i]}'.");

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
                            sd.Add(new Dictionary<string, long>());
                            td.Add(new Dictionary<string, long>());
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

#if DEBUG
            for (int j = 0; j < sd.Count; j++)
            {
                Logger.WriteLine(Logger.Level.debug, $"Original token size at group '{j}' source = '{sd[j].Count}' target = '{td[j].Count}'");
            }
#endif
            return (sd, td);
        }


        private (Dictionary<long, LinkedList<long>>, Dictionary<long, long>, string) BuildIndex()
        {
            Logger.WriteLine(Logger.Level.debug, $"Start to build index for data set.");

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
            Dictionary<long, long> len2lengths = new Dictionary<long, long>();

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

                    if (String.IsNullOrEmpty(rawSntPair.SrcSnt))
                    {
                        throw new InvalidDataException($"Source Line is empty. The data set is corrupted. SourceLine = '{rawSntPair.SrcSnt}', TargetLine = '{rawSntPair.TgtSnt}'");
                    }

                    if (String.IsNullOrEmpty(rawSntPair.TgtSnt))
                    {
                        throw new InvalidDataException($"Target Line is empty. The data set is corrupted. SourceLine = '{rawSntPair.SrcSnt}', TargetLine = '{rawSntPair.TgtSnt}'");
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
                    bw.Write(String.Join("\n", new string[] { rawSntPair.SrcSnt, rawSntPair.TgtSnt }));

                    long length = 0;
                    if (m_paddingEnums == PaddingEnums.NoPaddingInSrc)
                    {
                        length = rawSntPair.SrcGroupLenId;
                    }
                    else if (m_paddingEnums == PaddingEnums.NoPadding)
                    {
                        length = rawSntPair.GroupLenId;
                    }
                    else if (m_paddingEnums == PaddingEnums.NoPaddingInTgt)
                    {
                        length = rawSntPair.TgtGroupLenId;
                    }
                    else
                    {
                        // Completely random shuffle
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

                srSrc.Close();
                srTgt.Close();
            }

            bw.Close();

            Logger.WriteLine(Logger.Level.debug, $"Shuffled '{corpusSize}' sentence pairs.");

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
                //TODO(Zho): executed even if nothing is printed
                {
                    Logger.WriteLine(Logger.Level.debug, $"AggregateSrcLength = '{m_paddingEnums}'");
                    Logger.WriteLine(Logger.Level.debug, $"Src token length distribution");
                }

                int srcTotalNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcTotalNum += pair.Value;
                }

                int srcAccNum = 0;
                foreach (var pair in dictSrcLenDist)
                {
                    srcAccNum += pair.Value;

                    Logger.WriteLine(Logger.Level.debug, $"{pair.Key * 100} ~ {(pair.Key + 1) * 100}: {pair.Value} (acc: {100.0f * (float)srcAccNum / (float)srcTotalNum:F}%)");
                }

                Logger.WriteLine(Logger.Level.debug, $"Tgt token length distribution");

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

            return (len2offsets, len2lengths, binaryDataSetFilePath);
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
                (var length2offsets, var length2counts, string tmpDataSetFilePath) = BuildIndex();

                long totalRecordsNum = 0;
                foreach (var pair in length2offsets)
                {
                    totalRecordsNum += length2counts[pair.Key];
                }

                Logger.WriteLine(Logger.Level.debug, $"Start to sort and shuffle data set by length.");

                m_sortedIndexedDataSetFilePath = tmpDataSetFilePath + ".sorted";

#if DEBUG
                string tmp_sortedIndexedDataSetFilePath = tmpDataSetFilePath + ".sorted.txt";
                using (StreamWriter bwt = new StreamWriter(new FileStream(tmp_sortedIndexedDataSetFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 40960000)))
#endif
                using (BinaryWriter bw = new BinaryWriter(new FileStream(m_sortedIndexedDataSetFilePath, FileMode.Create, FileAccess.Write, FileShare.None, 40960000)))
                using (MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(tmpDataSetFilePath))
                using (MemoryMappedViewStream mms = mmf.CreateViewStream())
                {
                    using (BinaryReader br = new BinaryReader(mms))
                    {
                        while (length2offsets.Count > 0)
                        {
                            long length = GetNextLength(length2counts, totalRecordsNum);
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
                                length2counts[length]--;
                                totalRecordsNum--;

                                br.BaseStream.Seek(offset, SeekOrigin.Begin);

                                string[] srcTgtLine = br.ReadString().Split("\n");
                                string srcLine = srcTgtLine[0];
                                string tgtLine = srcTgtLine[1];

                                var srcTokens = srcLine.Split(' ').ToList();
                                var tgtTokens = tgtLine.Split(' ').ToList();

                                //if (srcTokens.Count > tgtTokens.Count)
                                //{
                                //    srcTokens = srcTokens.GetRange(0, tgtTokens.Count);
                                //    srcLine = String.Join(" ", srcTokens);
                                //}
                                //else
                                //{
                                //    tgtTokens = tgtTokens.GetRange(0, srcTokens.Count);
                                //    tgtLine = String.Join(" ", tgtTokens);
                                //}

                                totalSrcTokenSize += srcTokens.Count;
                                totalTgtTokenSize += tgtTokens.Count;

                                srcLines.Add(srcLine);
                                tgtLines.Add(tgtLine);


                                sentSize++;
                            }

                            bw.Write(sentSize);
                            bw.Write(String.Join("\n", srcLines));
                            bw.Write(String.Join("\n", tgtLines));

#if DEBUG
                            bwt.WriteLine(sentSize);
                            bwt.WriteLine(String.Join("\n", srcLines));
                            bwt.WriteLine(String.Join("\n", tgtLines));
#endif

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
                Logger.WriteLine(Logger.Level.err, $"Failed to prepare data set: '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            if (String.IsNullOrEmpty(m_sortedIndexedDataSetFilePath) || File.Exists(m_sortedIndexedDataSetFilePath) == false)
            {
                PrepareDataSet();
            }
            else
            {
                Logger.WriteLine(Logger.Level.debug, $"Use existing sorted indexed data set file '{m_sortedIndexedDataSetFilePath}'");
            }

            int batchIdx = 0;
            int currentBatchPercent = 0;
            MemoryMappedFile mmf = null;
            MemoryMappedViewStream mms = null;
            ZipDecompressor decompressor = null;
            if (m_sortedIndexedDataSetFilePath.ToLower().EndsWith(".zip"))
            {
                Logger.WriteLine($"The data set is a zip archive.");
                decompressor = new ZipDecompressor(m_sortedIndexedDataSetFilePath, m_dataPassword);
                mms = decompressor.GetMemoryMappedViewStream();
            }
            else
            {
                mmf = MemoryMappedFile.CreateFromFile(m_sortedIndexedDataSetFilePath);
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

                    string[] srcLines = br.ReadString().Split("\n");
                    string[] tgtLines = br.ReadString().Split("\n");
                    batchIdx++;

                    if (batchIdx < m_startBatchId)
                    {
                        continue;
                    }

                    if (batchIdx % 10000 == 0)
                    {
                        Logger.WriteLine(Logger.Level.debug, $"Processing batch '{batchIdx}'");
                    }

                    T batch;
                    int currentTokenCountsInBatch = 0;
                    for (int i = 0; i < sizeInBatch; i++)
                    {
                        var srcLine = srcLines[i];
                        var tgtLine = tgtLines[i];

                        if (m_batchNumInTotal > 0)
                        {
                            if ((100 * batchIdx / m_batchNumInTotal) > currentBatchPercent)
                            {
                                Logger.WriteLine($"Processing batch '{batchIdx}/{m_batchNumInTotal}'."); // The '{i}th' record in this batch is: Source = '{srcLine}' Target = '{tgtLine}'");
                                currentBatchPercent++;
                            }
                        }

                        IPair sntPair = new SntPair(srcLine, tgtLine);
                        currentTokenCountsInBatch += (sntPair.GetTgtTokenCount() + sntPair.GetSrcTokenCount());
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

            File.Delete(m_sortedIndexedDataSetFilePath);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public DPOPairCorpus()
        {

        }

        public DPOPairCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSrcSentLength = 32, int maxTgtSentLength = 32, PaddingEnums paddingEnums = PaddingEnums.AllowPadding, TooLongSequence tooLongSequence = TooLongSequence.Ignore, string indexedFilePath = null, int startBatchId = 0, string dataPassword = "")
        {
            Logger.WriteLine($"Loading parallel corpus from '{corpusFilePath}' for source side '{srcLangName}' and target side '{tgtLangName}' MaxSrcSentLength = '{maxSrcSentLength}',  MaxTgtSentLength = '{maxTgtSentLength}', Token Paading Type = '{paddingEnums}', TooLongSequence = '{tooLongSequence}', Encrypted data set = '{!String.IsNullOrEmpty(dataPassword)}'");
            m_maxTokenSizePerBatch = maxTokenSizePerBatch;
            m_maxSrcTokenSize = maxSrcSentLength;
            m_maxTgtTokenSize = maxTgtSentLength;
            m_tooLongSequence = tooLongSequence;
            m_paddingEnums = paddingEnums;
            CorpusName = corpusFilePath;
            m_sortedIndexedDataSetFilePath = indexedFilePath;
            m_dataPassword = dataPassword;

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
            m_startBatchId = startBatchId;
        }
    }
}
