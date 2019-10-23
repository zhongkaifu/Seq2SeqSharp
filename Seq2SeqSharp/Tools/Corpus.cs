using AdvUtils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        public int BatchSize { get { return SntPairs.Count; } }

        public SntPairBatch(List<SntPair> sntPairs)
        {
            SntPairs = sntPairs;
        }
    }

    public class Corpus : IEnumerable<SntPairBatch>
    {
        int m_maxSentLength = 32;
        int m_blockSize = 1000000;
        int m_batchSize = 1;

        List<string> m_srcFileList;
        List<string> m_tgtFileList;

        private const string m_srcShuffledFilePath = "shuffled.src.snt";
        private const string m_tgtShuffledFilePath = "shuffled.tgt.snt";

        public int CorpusSize = 0;

        public int BatchSize { get { return m_batchSize; } }

        public const string EOS = "<END>";
        public const string BOS = "<START>";


        void Shuffle(List<SntPair> sntPairs)
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
            Random rnd = new Random(DateTime.Now.Millisecond);
            foreach (KeyValuePair<int, List<SntPair>> pair in dict)
            {
                var sntPairList = pair.Value;
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
                if (pair.Value.Count < m_batchSize)
                {
                    //If the bucket size is less than batch size, ignore it
                    continue;
                }

                //Align the bucket size to batch size
                int externalItemCnt = pair.Value.Count % m_batchSize;
                pair.Value.RemoveRange(pair.Value.Count - externalItemCnt, externalItemCnt);

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

            foreach (var key in keys)
            {
                sntPairs.AddRange(sdict[key]);
            }
            
        }

        public void ShuffleAll(bool notShulledForExistingFiles)
        {
            if (File.Exists(m_srcShuffledFilePath) && File.Exists(m_tgtShuffledFilePath) && notShulledForExistingFiles)
            {
                Logger.WriteLine($"Shuffled files '{m_srcShuffledFilePath}' and '{m_tgtShuffledFilePath}' exist, so skip it. If they are not correct files, please delete them and rerun the command.");
                return;
            }

            Logger.WriteLine("Shuffling training corpus...");

            StreamWriter swSrc = new StreamWriter(m_srcShuffledFilePath, false);
            StreamWriter swTgt = new StreamWriter(m_tgtShuffledFilePath, false);

            List<SntPair> sntPairs = new List<SntPair>();
            CorpusSize = 0;
            var tooLongSntCnt = 0;
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
                
                    sntPair.SrcSnt = line.Split(new char[]{ ' '}, StringSplitOptions.RemoveEmptyEntries);

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
                        Logger.WriteLine($"Shuffle training corpus...");
                        Shuffle(sntPairs);
                        foreach (var item in sntPairs)
                        {
                            swSrc.WriteLine(String.Join(" ", item.SrcSnt));
                            swTgt.WriteLine(String.Join(" ", item.TgtSnt));
                        }
                        sntPairs.Clear();
                    }
                }

                srSrc.Close();
                srTgt.Close();
            }

            if (sntPairs.Count > 0)
            {
                Logger.WriteLine($"Shuffle training corpus...");
                Shuffle(sntPairs);
                foreach (var item in sntPairs)
                {
                    swSrc.WriteLine(String.Join(" ", item.SrcSnt));
                    swTgt.WriteLine(String.Join(" ", item.TgtSnt));
                }

                sntPairs.Clear();
            }


            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Shuffled '{CorpusSize}' sentence pairs.");
            Logger.WriteLine($"Found {tooLongSntCnt} sentences are longer than '{m_maxSentLength}' tokens, ignore them.");
        }

        public IEnumerator<SntPairBatch> GetEnumerator()
        {            
            ShuffleAll(true);

            StreamReader srSrc = new StreamReader(m_srcShuffledFilePath);
            StreamReader srTgt = new StreamReader(m_tgtShuffledFilePath);
            Random rnd = new Random(DateTime.Now.Millisecond);
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

                line = $"{BOS} {line.ToLower().Trim()} {EOS}";
                sntPair.SrcSnt = line.Split(' ');

                line = srTgt.ReadLine();
                sntPair.TgtSnt = line.ToLower().Trim().Split(' ');

                if ((lastSrcSntLen > 0 && lastSrcSntLen != sntPair.SrcSnt.Length) || outputs.Count > maxOutputsSize)
                {
                    // The length of current src sentence is different the previous one, so let's do mini shuffle and return it
                    for (int i = 0; i < outputs.Count; i++)
                    {
                        int idx = rnd.Next(0, outputs.Count);
                        var tmp = outputs[i];
                        outputs[i] = outputs[idx];
                        outputs[idx] = tmp;
                    }

                    //float y = ((float)outputs[0].SrcSnt.Length / (float)m_maxSentLength);
                    //int batchSize = Math.Max(1, m_batchSize / (int)Math.Pow(2, (int)(y * y)));
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

            srSrc.Close();
            srTgt.Close();

            for (int i = 0; i < outputs.Count; i++)
            {
                int idx = rnd.Next(0, outputs.Count);
                var tmp = outputs[i];
                outputs[i] = outputs[idx];
                outputs[idx] = tmp;
            }

            //float y2 = ((float)outputs[0].SrcSnt.Length / (float)m_maxSentLength);
            //int batchSize2 = Math.Max(1, m_batchSize / (int)Math.Pow(2, (int)(y2 * y2)));
            for (int i = 0; i < outputs.Count; i += m_batchSize)
            {
                int size = Math.Min(m_batchSize, outputs.Count - i);
                yield return new SntPairBatch(outputs.GetRange(i, size));
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Corpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 32)
        {
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSentLength = maxSentLength;

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
