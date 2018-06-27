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
        public List<string> SrcSnt;
        public List<string> TgtSnt;
    }

    public class Corpus : IEnumerable<SntPair>
    {
        const int BLOCKSIZE = 100000;

        List<string> srcFileList;
        List<string> tgtFileList;

        void Shuffle(List<SntPair> sntPairs)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < sntPairs.Count; i++)
            {
                int idx = rnd.Next(0, sntPairs.Count);
                SntPair tmp = sntPairs[i];
                sntPairs[i] = sntPairs[idx];
                sntPairs[idx] = tmp;
            }
        }

        public IEnumerator<SntPair> GetEnumerator()
        {
            for (int i = 0; i < srcFileList.Count; i++)
            {
                StreamReader srSrc = new StreamReader(srcFileList[i]);
                StreamReader srTgt = new StreamReader(tgtFileList[i]);

                List<SntPair> sntPairs = new List<SntPair>();

                while (true)
                {
                    string line;
                    SntPair sntPair = new SntPair();
                    if ((line = srSrc.ReadLine()) == null)
                    {
                        break;
                    }
                    sntPair.SrcSnt = line.ToLower().Trim().Split(' ').ToList();

                    line = srTgt.ReadLine();
                    sntPair.TgtSnt = line.ToLower().Trim().Split(' ').ToList();

                  
                    sntPairs.Add(sntPair);
                    if (sntPairs.Count >= BLOCKSIZE)
                    {
                        Console.WriteLine($"Shuffle training corpus...");
                        Shuffle(sntPairs);
                        foreach (var item in sntPairs)
                        {
                            yield return item;
                        }
                        sntPairs.Clear();
                    }
                }

                if (sntPairs.Count > 0)
                {
                    Shuffle(sntPairs);
                    foreach (var item in sntPairs)
                    {
                        yield return item;
                    }
                    sntPairs.Clear();
                }

                srSrc.Close();
                srTgt.Close();
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public Corpus(string corpusFilePath, string srcLangName, string tgtLangName)
        {
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
