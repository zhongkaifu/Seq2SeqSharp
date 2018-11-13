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
        int blockSize = 1000000;

        List<string> srcFileList;
        List<string> tgtFileList;

        private const string srcShuffledFilePath = "shuffled.src.snt";
        private const string tgtShuffledFilePath = "shuffled.tgt.snt";

        public int CorpusSize = 0;

        void Shuffle(SntPair[] sntPairs)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < sntPairs.Length; i++)
            {
                int idx = rnd.Next(0, sntPairs.Length);
                SntPair tmp = sntPairs[i];
                sntPairs[i] = sntPairs[idx];
                sntPairs[idx] = tmp;
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


                    sntPairs.Add(sntPair);
                    CorpusSize++;
                    if (blockSize > 0 && sntPairs.Count >= blockSize)
                    {
                        SntPair[] arraySntPairs = sntPairs.ToArray();
                        sntPairs = new List<SntPair>();
                        GC.Collect();

                        Logger.WriteLine($"Shuffle training corpus...");
                        Shuffle(arraySntPairs);
                        foreach (var item in arraySntPairs)
                        {
                            swSrc.WriteLine(String.Join(" ", item.SrcSnt));
                            swTgt.WriteLine(String.Join(" ", item.TgtSnt));
                        }

                    }
                }

                srSrc.Close();
                srTgt.Close();
            }

            if (sntPairs.Count > 0)
            {
                SntPair[] arraySntPairs = sntPairs.ToArray();
                sntPairs = new List<SntPair>();
                GC.Collect();

                Logger.WriteLine($"Shuffle training corpus...");
                Shuffle(arraySntPairs);
                foreach (var item in arraySntPairs)
                {
                    swSrc.WriteLine(String.Join(" ", item.SrcSnt));
                    swTgt.WriteLine(String.Join(" ", item.TgtSnt));
                }
            }


            swSrc.Close();
            swTgt.Close();
        }

        public IEnumerator<SntPair> GetEnumerator()
        {
            ShuffleAll(true);

            StreamReader srSrc = new StreamReader(srcShuffledFilePath);
            StreamReader srTgt = new StreamReader(tgtShuffledFilePath);

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

                yield return sntPair;
            }

            srSrc.Close();
            srTgt.Close();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Corpus(string corpusFilePath, string srcLangName, string tgtLangName, int shuffleBlockSize = -1)
        {
            blockSize = shuffleBlockSize;

            srcFileList = new List<string>();
            tgtFileList = new List<string>();
            string[] srcFiles = Directory.GetFiles(corpusFilePath, $"*.{srcLangName}.snt", SearchOption.TopDirectoryOnly);
            foreach (string srcFile in srcFiles)
            {
                string tgtFile = srcFile.Replace($".{srcLangName}.", $".{tgtLangName}.");

                srcFileList.Add(srcFile);
                tgtFileList.Add(tgtFile);
            }
        }
    }
}
