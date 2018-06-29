using Seq2SeqSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using Seq2SeqSharp.Tools;

namespace Seq2SeqConsole
{
    class Program
    {
        static List<string> logLines = new List<string>();
        static object locker = new object();
        static string DefaultEncodedModelFilePath = "Seq2Seq.model";

        static void ss_IterationDone(object sender, EventArgs e)
        {
            CostEventArg ep = e as CostEventArg;

            if (float.IsInfinity(ep.Cost) == false)
            {
                TimeSpan ts = DateTime.Now - ep.StartDateTime;
                double sentPerMin = 0;
                if (ts.TotalMinutes > 0)
                {
                    sentPerMin = ep.ProcessedInTotal / ts.TotalMinutes;
                }

                lock (locker)
                {
                    string logLine = $"{DateTime.Now}: Epoch = '{ep.Epoch}' Learning Rate = '{ep.LearningRate}', Avg Cost = '{ep.CostInTotal / ep.ProcessedInTotal}', SentInTotal = '{ep.ProcessedInTotal}', SentPerMin = '{sentPerMin}'";

                    logLines.Add(logLine);

                    Console.WriteLine(logLine);

                    if (logLines.Count >= 100)
                    {
                        File.AppendAllLines("logs.txt", logLines);
                        logLines.Clear();
                    }
                }
            }

        }

        private static int ArgPos(string str, string[] args)
        {
            str = str.ToLower();
            for (var a = 0; a < args.Length; a++)
            {
                if (str == args[a].ToLower())
                {
                    if (a == args.Length - 1)
                    {
                        return -1;
                    }
                    return a;
                }
            }
            return -1;
        }

        static void UsageTrain()
        {
            Console.WriteLine("Seq2SeqConsole.exe train [parameters...]");
            Console.WriteLine("Parameters:");
            Console.WriteLine("-WordVectorSize: The vector size of encoded source word.");
            Console.WriteLine("-HiddenSize: The hidden layer size of encoder. The hidden layer size of decoder is 2x (-HiddenSize)");
            Console.WriteLine("-LearningRate: Learning rate. Default value is 0.001");
            Console.WriteLine("-Depth: The network depth in decoder. Default value is 1");
            Console.WriteLine("-ModelFilePath: The trained model file path.");
            Console.WriteLine("-SrcVocab: The vocabulary file path for source side.");
            Console.WriteLine("-TgtVocab: The vocabulary file path for target side.");
            Console.WriteLine("-SrcLang: Source language name.");
            Console.WriteLine("-TgtLang: Target language name.");
            Console.WriteLine("-TrainCorpusPath: training corpus folder path");
            Console.WriteLine("-UseSparseFeature: It indicates if sparse feature used for training.");
        }

        static void UsageTest()
        {
            Console.WriteLine("Seq2SeqConsole.exe predict [parameters...]");
            Console.WriteLine("Parameters:");
            Console.WriteLine("-InputTestFile: The input file for test.");
            Console.WriteLine("-OutputTestFile: The test result file.");
            Console.WriteLine("-ModelFilePath: The trained model file path.");
        }


        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Seq2SeqConsole.exe [train|predict] [parameters...]");
                return;
            }

            AttentionSeq2Seq ss = null;
            string modelFilePath = DefaultEncodedModelFilePath;
            int i = 0;
            if ((i = ArgPos("-ModelFilePath", args)) >= 0) modelFilePath = args[i + 1];

           
            if (args[0] == "train")
            {            
                int wordVectorSize = 200;
                int hiddenSize = 200;
                string srcLangName = String.Empty;
                string tgtLangName = String.Empty;
                float learningRate = 0.001f;
                int depth = 1;
                string srcVocabFilePath = null;
                string tgtVocabFilePath = null;
                string sntTrainCorpusPath = String.Empty;
                bool useSparseFeature = true;

                if ((i = ArgPos("-WordVectorSize", args)) >= 0) wordVectorSize = int.Parse(args[i + 1]);
                if ((i = ArgPos("-HiddenSize", args)) >= 0) hiddenSize = int.Parse(args[i + 1]);
                if ((i = ArgPos("-LearningRate", args)) >= 0) learningRate = float.Parse(args[i + 1]);
                if ((i = ArgPos("-Depth", args)) >= 0) depth = int.Parse(args[i + 1]);
                if ((i = ArgPos("-SrcVocab", args)) >= 0) srcVocabFilePath = args[i + 1];
                if ((i = ArgPos("-TgtVocab", args)) >= 0) tgtVocabFilePath = args[i + 1];
                if ((i = ArgPos("-SrcLang", args)) >= 0) srcLangName = args[i + 1];
                if ((i = ArgPos("-TgtLang", args)) >= 0) tgtLangName = args[i + 1];
                if ((i = ArgPos("-TrainCorpusPath", args)) >= 0) sntTrainCorpusPath = args[i + 1];
                if ((i = ArgPos("-UseSparseFeature", args)) >= 0) useSparseFeature = bool.Parse(args[i + 1]);

                Corpus trainCorpus = new Corpus(sntTrainCorpusPath, srcLangName, tgtLangName);
                if (File.Exists(modelFilePath) == false)
                {
                    ss = new AttentionSeq2Seq(wordVectorSize, hiddenSize, depth, trainCorpus, srcVocabFilePath, tgtVocabFilePath, useSparseFeature, true, modelFilePath);
                }
                else
                {
                    Console.WriteLine($"Loading model from '{modelFilePath}'...");
                    ss = new AttentionSeq2Seq();
                    ss.Load(modelFilePath);
                    ss.TrainCorpus = trainCorpus;
                }

                Console.WriteLine($"Source Language = '{srcLangName}'");
                Console.WriteLine($"Target Language = '{tgtLangName}'");
                Console.WriteLine($"SSE Enable = '{System.Numerics.Vector.IsHardwareAccelerated}'");
                Console.WriteLine($"SSE Size = '{System.Numerics.Vector<float>.Count * 32}'");
                Console.WriteLine($"Processor counter = '{Environment.ProcessorCount}'");
                Console.WriteLine($"Hidden Size = '{hiddenSize}'");
                Console.WriteLine($"Word Vector Size = '{wordVectorSize}'");
                Console.WriteLine($"Learning Rate = '{learningRate}'");
                Console.WriteLine($"Network Layer = '{depth}'");
                Console.WriteLine($"Use Sparse Feature = '{useSparseFeature}'");

                ss.IterationDone += ss_IterationDone;
                ss.Train(300, learningRate);
            }
            else if (args[0] == "predict")
            {
                string inputTestFile = String.Empty;
                string outputTestFile = String.Empty;

                if ((i = ArgPos("-InputTestFile", args)) >= 0) inputTestFile = args[i + 1];
                if ((i = ArgPos("-OutputTestFile", args)) >= 0) outputTestFile = args[i + 1];

                ss = new AttentionSeq2Seq();
                ss.Load(modelFilePath);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(inputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    List<string> outputWords = ss.Predict(line.ToLower().Trim().Split(' ').ToList());
                    outputLines.Add(String.Join(" ", outputWords));
                }

                File.WriteAllLines(outputTestFile, outputLines);
            }
        }       
    }
}
