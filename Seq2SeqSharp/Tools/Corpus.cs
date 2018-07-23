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
        int blockSize = 100000;

        List<string> srcFileList;
        List<string> tgtFileList;

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

        public IEnumerator<SntPair> GetEnumerator()
        {
            List<SntPair> sntPairs = new List<SntPair>();
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
                    if (blockSize > 0 && sntPairs.Count >= blockSize)
                    {
                        SntPair[] arraySntPairs = sntPairs.ToArray();
                        sntPairs = new List<SntPair>();
                        GC.Collect();

                        Logger.WriteLine($"Shuffle training corpus...");
                        Shuffle(arraySntPairs);
                        foreach (var item in arraySntPairs)
                        {
                            yield return item;
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
                    yield return item;
                }
            }
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
