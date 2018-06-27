using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FilterLowFreqTerm
{
    class Program
    {
        static bool IsEnglishDigit(string word)
        {
            word = word.ToLower();
            for (int i = 0; i < word.Length; i++)
            {
                if (word[i] >= 'a' && word[i] <= 'z')
                {
                    return true;
                }

                if (word[i] >= '0' && word[i] <= '9')
                {
                    return true;
                }
            }

            return false;
        }

        static Dictionary<string, long> FilterOutPureEnglishDigitNumber(Dictionary<string, long> vocab)
        {
            Dictionary<string, long> newVocab = new Dictionary<string, long>();
            foreach (var pair in vocab)
            {
                if (IsEnglishDigit(pair.Key) == true && pair.Value < 1000)
                {
                    continue;
                }

                newVocab.Add(pair.Key, pair.Value);
            }

            return newVocab;
        }

        static void Main(string[] args)
        {
            if (args.Length != 5)
            {
                Console.WriteLine($"FilterLowFreqTerm.exe [Corpus File Path] [Source Language Name] [Target Language Name] [Min Term Freq] [Output Corpus File Name]");
                return;
            }

            string inputCorpusFilePath = args[0];
            string srcLangName = args[1];
            string tgtLangName = args[2];
            int minTermFreq = int.Parse(args[3]);
            string outputCorpusFileName = args[4];

            List<List<string>> srcCorpus = new List<List<string>>();
            List<List<string>> tgtCorpus = new List<List<string>>();

            LoadTrainingCorpus(srcCorpus, tgtCorpus, srcLangName, tgtLangName, inputCorpusFilePath);

            Dictionary<string, long> srcWord2Cnt = new Dictionary<string, long>();
            Dictionary<string, long> tgtWord2Cnt = new Dictionary<string, long>();

            foreach (var sent in srcCorpus)
            {
                foreach (var word in sent)
                {
                    if (srcWord2Cnt.ContainsKey(word) == false)
                    {
                        srcWord2Cnt.Add(word, 0);
                    }
                    srcWord2Cnt[word]++;
                }
            }

            foreach (var sent in tgtCorpus)
            {
                foreach (var word in sent)
                {
                    if (tgtWord2Cnt.ContainsKey(word) == false)
                    {
                        tgtWord2Cnt.Add(word, 0);
                    }
                    tgtWord2Cnt[word]++;
                }
            }


            tgtWord2Cnt = FilterOutPureEnglishDigitNumber(tgtWord2Cnt);

            List<string> newSrcCorpus = new List<string>();
            List<string> newTgtCorpus = new List<string>();

            for (int i = 0; i < srcCorpus.Count; i++)
            {
                List<string> srcSnt = srcCorpus[i];
                List<string> tgtSnt = tgtCorpus[i];

                if (ShouldRemoved(srcSnt, srcWord2Cnt, minTermFreq) == false &&
                    ShouldRemoved(tgtSnt, tgtWord2Cnt, minTermFreq) == false)
                {
                    newSrcCorpus.Add(String.Join(" ", srcSnt));
                    newTgtCorpus.Add(String.Join(" ", tgtSnt));
                }
            }

            Console.WriteLine($"{newSrcCorpus.Count} sentence pairs saved.");                   

            File.WriteAllLines($"{outputCorpusFileName}.{srcLangName}.snt", newSrcCorpus);
            File.WriteAllLines($"{outputCorpusFileName}.{tgtLangName}.snt", newTgtCorpus);

            srcCorpus.Clear();
            tgtCorpus.Clear();

            SortedDictionary<long, List<string>> sdict = new SortedDictionary<long, List<string>>();
            foreach (var pair in srcWord2Cnt)
            {
                if (sdict.ContainsKey(pair.Value) == false)
                {
                    sdict.Add(pair.Value, new List<string>());
                }
                sdict[pair.Value].Add(pair.Key);
            }


            List<string> vocabList = new List<string>();
            foreach (var pair in sdict.Reverse())
            {
                if (pair.Key >= minTermFreq)
                {
                    foreach (string word in pair.Value)
                    {
                        vocabList.Add($"{word}\t{pair.Key}");
                    }
                }
            }
            File.WriteAllLines("SrcVocab.txt", vocabList);
            Console.WriteLine($"Src vocab size = '{vocabList.Count}'");

            sdict.Clear();
            vocabList.Clear();

            foreach (var pair in tgtWord2Cnt)
            {
                if (sdict.ContainsKey(pair.Value) == false)
                {
                    sdict.Add(pair.Value, new List<string>());
                }
                sdict[pair.Value].Add(pair.Key);
            }

            vocabList = new List<string>();
            foreach (var pair in sdict.Reverse())
            {
                if (pair.Key >= minTermFreq)
                {
                    foreach (string word in pair.Value)
                    {
                        vocabList.Add($"{word}\t{pair.Key}");
                    }
                }
            }
            File.WriteAllLines("TgtVocab.txt", vocabList);
            Console.WriteLine($"Tgt vocab size = '{vocabList.Count}'");
        }


        private static bool ShouldRemoved(List<string> snt, Dictionary<string, long> word2Cnt, int minFreq)
        {
            foreach (string word in snt)
            {
                if (word2Cnt.ContainsKey(word) == false)
                {
                    return true;
                }

                if (word2Cnt[word] < minFreq)
                {
                    return true;
                }
            }

            return false;
        }


        private static void LoadTrainingCorpus(List<List<string>> input, List<List<string>> output, string srcLangName, string tgtLangName, string sntCorpusPath)
        {
            HashSet<string> setSnt = new HashSet<string>();

            string[] srcFiles = Directory.GetFiles(sntCorpusPath, $"*.{srcLangName}.snt", SearchOption.TopDirectoryOnly);
            long sntCnt = 0;

            foreach (string srcFile in srcFiles)
            {
                string tgtFile = srcFile.Replace($".{srcLangName}.", $".{tgtLangName}.");

                Console.WriteLine($"Loading training corpus file path = '{srcFile}' and '{tgtFile}'");

                var data_sents_raw1 = File.ReadAllLines(srcFile);
                var data_sents_raw2 = File.ReadAllLines(tgtFile);

                for (int k = 0; k < data_sents_raw1.Length; k++)
                {
                    string s = $"{data_sents_raw1[k].ToLower().Trim()}\t{data_sents_raw2[k].ToLower().Trim()}";

                    if (setSnt.Contains(s) == false)
                    {
                        input.Add(data_sents_raw1[k].ToLower().Trim().Split(' ').ToList());
                        output.Add(data_sents_raw2[k].ToLower().Trim().Split(' ').ToList());

                        sntCnt++;

                        setSnt.Add(s);
                    }
                }

                setSnt.Clear();
            }

            Console.WriteLine($"{sntCnt} sentence pairs loaded.");
        }
    }
}
