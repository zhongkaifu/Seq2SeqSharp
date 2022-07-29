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
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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
        internal int m_maxSrcSentLength = 32;
        internal int m_maxTgtSentLength = 32;
        internal int m_blockSize = 1000000;
        internal int m_batchSize = 1;
        internal List<string> m_srcFileList;
        internal List<string> m_tgtFileList;
        internal ShuffleEnums m_shuffleEnums;

        public int BatchSize => m_batchSize;        
        private bool m_showTokenDist = true;

        private readonly Random rnd = new Random(DateTime.Now.Millisecond);

        public string CorpusName { get; set; }

        private TooLongSequence m_tooLongSequence = TooLongSequence.Ignore;

        private string binaryDataSetFilePath = "";

        private Dictionary<long, LinkedList<long>> BuildIndex()
        {
            SortedDictionary<int, int> dictSrcLenDist = new SortedDictionary<int, int>();
            SortedDictionary<int, int> dictTgtLenDist = new SortedDictionary<int, int>();
            int corpusSize = 0;
            int tooLongSrcSntCnt = 0;
            int tooLongTgtSntCnt = 0;
            string randomFileName = Path.GetRandomFileName();
            Logger.WriteLine($"Loading and shuffling corpus from '{m_srcFileList.Count}' files.");

            binaryDataSetFilePath = randomFileName + ".tmp";
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

                    RawSntPair rawSntPair = new RawSntPair(srSrc.ReadLine(), srTgt.ReadLine(), m_maxSrcSentLength, m_maxTgtSentLength, m_tooLongSequence == TooLongSequence.Truncation);
                    if (rawSntPair.IsEmptyPair())
                    {
                        break;
                    }

                    if (m_showTokenDist)
                    {
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
                    }

                    bool hasTooLongSent = false;
                    if (rawSntPair.SrcLength > m_maxSrcSentLength)
                    {
                        Interlocked.Increment(ref tooLongSrcSntCnt);
                        hasTooLongSent = true;
                    }

                    if (rawSntPair.TgtLength > m_maxTgtSentLength)
                    {
                        Interlocked.Increment(ref tooLongTgtSntCnt);
                        hasTooLongSent = true;
                    }

                    if (hasTooLongSent)
                    {
                        continue;
                    }

                    long offset = bw.BaseStream.Position;                                      
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

            return len2offsets;
        }
        

        public IEnumerator<T> GetEnumerator()
        {
            var length2offsets = BuildIndex();
            
            using(BinaryReader br = new BinaryReader(new FileStream(binaryDataSetFilePath, FileMode.Open)))
            {
                int maxOutputsSize = m_batchSize * 10000;
                List<SntPair> outputs = new List<SntPair>();

                int lengthRnd = rnd.Next(length2offsets.Count);
                long length = length2offsets.Keys.ToArray()[lengthRnd];
                LinkedList<long> offsets = length2offsets[length];
                bool isAbort = false;

                while (length2offsets.Count > 0)
                {
                    bool lengthChanged = false;
                    while (offsets.Count == 0)
                    {
                        length2offsets.Remove(length);
                        if (length2offsets.Count == 0)
                        {
                            isAbort = true;
                            break;
                        }

                        lengthRnd = rnd.Next(length2offsets.Count);
                        length = length2offsets.Keys.ToArray()[lengthRnd];
                        offsets = length2offsets[length];
                        lengthChanged = true;
                    }

                    if (isAbort)
                    {
                        break;
                    }

                    if (outputs.Count > maxOutputsSize || lengthChanged == true)
                    {
                        for (int i = 0; i < outputs.Count; i += m_batchSize)
                        {
                            int size = Math.Min(m_batchSize, outputs.Count - i);
                            var batch = new T();
                            batch.CreateBatch(outputs.GetRange(i, size));
                            yield return batch;
                        }

                        outputs.Clear();

                        //Force to select sequences with different length
                        lengthRnd = rnd.Next(length2offsets.Count);
                        length = length2offsets.Keys.ToArray()[lengthRnd];
                        offsets = length2offsets[length];
                    }
                    else
                    {
                        long offset = offsets.First.Value;
                        offsets.RemoveFirst();

                        br.BaseStream.Seek(offset, SeekOrigin.Begin);
                        var srcLine = br.ReadString();
                        var tgtLine = br.ReadString();
                        SntPair sntPair = new SntPair(srcLine, tgtLine);
                        outputs.Add(sntPair);
                    }
                }

                for (int i = 0; i < outputs.Count; i += m_batchSize)
                {
                    int size = Math.Min(m_batchSize, outputs.Count - i);
                    var batch = new T();
                    batch.CreateBatch(outputs.GetRange(i, size));
                    yield return batch;
                }
            }

            File.Delete(binaryDataSetFilePath);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public ParallelCorpus()
        {

        }

        public ParallelCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
        {
            Logger.WriteLine($"Loading parallel corpus from '{corpusFilePath}' for source side '{srcLangName}' and target side '{tgtLangName}' MaxSrcSentLength = '{maxSrcSentLength}',  MaxTgtSentLength = '{maxTgtSentLength}', aggregateSrcLengthForShuffle = '{shuffleEnums}', TooLongSequence = '{tooLongSequence}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSrcSentLength;
            m_maxTgtSentLength = maxTgtSentLength;

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
