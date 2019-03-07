using Seq2SeqSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using Seq2SeqSharp.Tools;
using AdvUtils;
using TensorSharp;

namespace Seq2SeqConsole
{
    class Program
    {
        static void ss_IterationDone(object sender, EventArgs e)
        {
            CostEventArg ep = e as CostEventArg;

            if (float.IsInfinity(ep.CostPerWord) == false)
            {
                TimeSpan ts = DateTime.Now - ep.StartDateTime;
                double sentPerMin = 0;
                double wordPerSec = 0;
                if (ts.TotalMinutes > 0)
                {
                    sentPerMin = ep.ProcessedSentencesInTotal / ts.TotalMinutes;
                }

                if (ts.TotalSeconds > 0)
                {
                    wordPerSec = ep.ProcessedWordsInTotal / ts.TotalSeconds;
                }

                Logger.WriteLine($"Epoch = '{ep.Epoch}' LR = '{ep.AvgLearningRate}', Current Cost = '{ep.CostPerWord.ToString("F6")}', Avg Cost = '{ep.avgCostInTotal.ToString("F6")}', SentInTotal = '{ep.ProcessedSentencesInTotal}', SentPerMin = '{sentPerMin.ToString("F")}', WordPerSec = '{wordPerSec.ToString("F")}'");
            }

        }

        public static String GetTimeStamp(DateTime timeStamp)
        {
            return String.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }

        static void Main(string[] args)
        {
            Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{GetTimeStamp(DateTime.Now)}.log";

            Options options = new Options();
            ArgParser argParser = new ArgParser(args, options);
            ShowOptions(args, options);

            AttentionSeq2Seq ss = null;
            ArchTypeEnums archType = (ArchTypeEnums)options.ArchType;

            //Parse device ids from options
            string[] deviceIdsStr = options.DeviceIds.Split(',');
            int[] deviceIds = new int[deviceIdsStr.Length];
            for (int i = 0; i < deviceIdsStr.Length; i++)
            {
                deviceIds[i] = int.Parse(deviceIdsStr[i]);
            }

            if (String.Equals(options.TaskName, "train", StringComparison.InvariantCultureIgnoreCase))
            {
                Corpus trainCorpus = new Corpus(options.TrainCorpusPath, options.SrcLang, options.TgtLang, options.BatchSize * deviceIds.Length, options.ShuffleBlockSize);
                if (File.Exists(options.ModelFilePath) == false)
                {
                    //New training
                    ss = new AttentionSeq2Seq(options.WordVectorSize, options.HiddenSize, options.Depth, trainCorpus, options.SrcVocab, options.TgtVocab, options.SrcEmbeddingModelFilePath, options.TgtEmbeddingModelFilePath,
                        true, options.ModelFilePath, options.BatchSize, options.DropoutRatio, archType, deviceIds);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{options.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(options.ModelFilePath, options.BatchSize, archType, deviceIds);
                    ss.TrainCorpus = trainCorpus;
                }

                ss.IterationDone += ss_IterationDone;
                ss.Train(100, options.LearningRate, options.GradClip);
            }
            else if (String.Equals(options.TaskName, "test", StringComparison.InvariantCultureIgnoreCase))
            {
                //Test trained model
                ss = new AttentionSeq2Seq(options.ModelFilePath, 1, archType, deviceIds);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(options.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    List<List<string>> outputWordsList = ss.Predict(line.ToLower().Trim().Split(' ').ToList(), options.BeamSearch);
                    foreach (var outputWords in outputWordsList)
                    {
                        outputLines.Add(String.Join(" ", outputWords));
                    }
                }

                File.WriteAllLines(options.OutputTestFile, outputLines);
            }
            else
            {
                argParser.Usage();
            }
        }

        private static void ShowOptions(string[] args, Options options)
        {
            string commandLine = String.Join(" ", args);
            Logger.WriteLine($"Command Line = '{commandLine}'");

            ArchTypeEnums archType = (ArchTypeEnums)options.ArchType;

            Logger.WriteLine($"Source Language = '{options.SrcLang}'");
            Logger.WriteLine($"Target Language = '{options.TgtLang}'");
            Logger.WriteLine($"SSE Enable = '{System.Numerics.Vector.IsHardwareAccelerated}'");
            Logger.WriteLine($"SSE Size = '{System.Numerics.Vector<float>.Count * 32}'");
            Logger.WriteLine($"Processor counter = '{Environment.ProcessorCount}'");
            Logger.WriteLine($"Hidden Size = '{options.HiddenSize}'");
            Logger.WriteLine($"Word Vector Size = '{options.WordVectorSize}'");
            Logger.WriteLine($"Learning Rate = '{options.LearningRate}'");
            Logger.WriteLine($"Network Layer = '{options.Depth}'");
            Logger.WriteLine($"Gradient Clip = '{options.GradClip}'");
            Logger.WriteLine($"Dropout Ratio = '{options.DropoutRatio}'");
            Logger.WriteLine($"Batch Size = '{options.BatchSize}'");
            Logger.WriteLine($"Arch Type = '{archType}'");
            Logger.WriteLine($"Device Ids = '{options.DeviceIds}'");
        }
    }
}
