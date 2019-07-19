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

    public class Corpus : IEnumerable<SntPair>
    {
        int maxSentLength = 32;
        int blockSize = 1000000;
        int batchSize = 1;

        List<string> srcFileList;
        List<string> tgtFileList;

        private const string srcShuffledFilePath = "shuffled.src.snt";
        private const string tgtShuffledFilePath = "shuffled.tgt.snt";

        public int CorpusSize = 0;

        public int BatchSize { get { return batchSize; } }

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
                if (pair.Value.Count < batchSize)
                {
                    //If the bucket size is less than batch size, ignore it
                    continue;
                }

                //Align the bucket size to batch size
                int externalItemCnt = pair.Value.Count % batchSize;
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
            Logger.WriteLine("Shuffling training corpus...");

            if (File.Exists(srcShuffledFilePath) && File.Exists(tgtShuffledFilePath) && notShulledForExistingFiles)
            {
                return;
            }


            StreamWriter swSrc = new StreamWriter(srcShuffledFilePath, false);
            StreamWriter swTgt = new StreamWriter(tgtShuffledFilePath, false);

            List<SntPair> sntPairs = new List<SntPair>();
            CorpusSize = 0;
            var tooLongSntCnt = 0;
            for (int i = 0; i < srcFileList.Count; i++)
            {
                StreamReader srSrc = new StreamReader(srcFileList[i]);
                StreamReader srTgt = new StreamReader(tgtFileList[i]);

                while (true)
                {
                    string line;
                    SntPair sntPair = new SntPair();
                    if ((line = srSrc.ReadLine()) == null)
                    {
                        break;
                    }

                    sntPair.SrcSnt = line.ToLower().Trim().Split(' ').ToArray();

                    line = srTgt.ReadLine();
                    sntPair.TgtSnt = line.ToLower().Trim().Split(' ').ToArray();

                    if (sntPair.SrcSnt.Length >= maxSentLength || sntPair.TgtSnt.Length >= maxSentLength)
                    {
                        tooLongSntCnt++;
                        continue;
                    }

                    sntPairs.Add(sntPair);
                    CorpusSize++;
                    if (blockSize > 0 && sntPairs.Count >= blockSize)
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
            Logger.WriteLine($"Found {tooLongSntCnt} sentences are longer than '{maxSentLength}' tokens, ignore them.");
        }

        public IEnumerator<SntPair> GetEnumerator()
        {
            ShuffleAll(true);

            StreamReader srSrc = new StreamReader(srcShuffledFilePath);
            StreamReader srTgt = new StreamReader(tgtShuffledFilePath);
            Random rnd = new Random(DateTime.Now.Millisecond);
            int lastSrcSntLen = -1;
            const int maxOutputsSize = 1000000;
            List<SntPair> outputs = new List<SntPair>();

            while (true)
            {
                string line;
                SntPair sntPair = new SntPair();
                if ((line = srSrc.ReadLine()) == null)
                {
                    break;
                }
                sntPair.SrcSnt = line.ToLower().Trim().Split(' ').ToArray();

                line = srTgt.ReadLine();
                sntPair.TgtSnt = line.ToLower().Trim().Split(' ').ToArray();

                if ((lastSrcSntLen > 0 && lastSrcSntLen != sntPair.SrcSnt.Length) || outputs.Count > maxOutputsSize)
                {
                    for (int i = 0; i < outputs.Count; i++)
                    {
                        int idx = rnd.Next(0, outputs.Count);
                        var tmp = outputs[i];
                        outputs[i] = outputs[idx];
                        outputs[idx] = tmp;
                    }

                    foreach (var sntPairItem in outputs)
                    {
                        yield return sntPairItem;
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

            foreach (var sntPairItem in outputs)
            {
                yield return sntPairItem;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Corpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 32)
        {
            this.batchSize = batchSize;
            blockSize = shuffleBlockSize;
            this.maxSentLength = maxSentLength;

            srcFileList = new List<string>();
            tgtFileList = new List<string>();
            string[] srcFiles = Directory.GetFiles(corpusFilePath, $"*.{srcLangName}.snt", SearchOption.TopDirectoryOnly);
            foreach (string srcFile in srcFiles)
            {
                string tgtFile = srcFile.ToLower().Replace($".{srcLangName.ToLower()}.", $".{tgtLangName.ToLower()}.");

                srcFileList.Add(srcFile);
                tgtFileList.Add(tgtFile);
            }
        }
    }
}
