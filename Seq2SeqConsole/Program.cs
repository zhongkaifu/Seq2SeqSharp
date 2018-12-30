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
                if (ts.TotalMinutes > 0)
                {
                    sentPerMin = ep.ProcessedInTotal / ts.TotalMinutes;
                }
                Logger.WriteLine($"Epoch = '{ep.Epoch}' LR = '{ep.AvgLearningRate}', Current Cost = '{ep.CostPerWord.ToString("F6")}', Avg Cost = '{ep.avgCostInTotal.ToString("F6")}', SentInTotal = '{ep.ProcessedInTotal}', SentPerMin = '{sentPerMin.ToString("F")}'");
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

        public static String GetTimeStamp(DateTime timeStamp)
        {
            return String.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }

        static void Main(string[] args)
        {
            //float[] f = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
            //Tensor t = Tensor.FromArray(TensorAllocator.Allocator, f).View(12, 1);

            //Tensor t2 = t.Expand(12, 4);

            //Tensor mt = Ops.Max(null, t, 1);


            //Tensor t2 = t.Unfold(0, 2, 2);




            Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{GetTimeStamp(DateTime.Now)}.log";

            Options options = new Options();
            ArgParser argParser = new ArgParser(args, options);

            AttentionSeq2Seq ss = null;
            ArchTypeEnums archType = (ArchTypeEnums)options.ArchType;

            if (String.Equals(options.TaskName, "train", StringComparison.InvariantCultureIgnoreCase))
            {
                Corpus trainCorpus = new Corpus(options.TrainCorpusPath, options.SrcLang, options.TgtLang, options.BatchSize, options.ShuffleBlockSize);
                if (File.Exists(options.ModelFilePath) == false)
                {
                    ss = new AttentionSeq2Seq(options.WordVectorSize, options.HiddenSize, options.Depth, trainCorpus, options.SrcVocab, options.TgtVocab, options.SrcEmbeddingModelFilePath, options.TgtEmbeddingModelFilePath,
                        true, options.ModelFilePath, options.BatchSize, options.DropoutRatio, archType);
                }
                else
                {
                    Logger.WriteLine($"Loading model from '{options.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(options.ModelFilePath, options.BatchSize, archType);
                    ss.TrainCorpus = trainCorpus;
                }

                Logger.WriteLine($"Source Language = '{options.SrcLang}'");
                Logger.WriteLine($"Target Language = '{options.TgtLang}'");
                Logger.WriteLine($"SSE Enable = '{System.Numerics.Vector.IsHardwareAccelerated}'");
                Logger.WriteLine($"SSE Size = '{System.Numerics.Vector<float>.Count * 32}'");
                Logger.WriteLine($"Processor counter = '{Environment.ProcessorCount}'");
                Logger.WriteLine($"Hidden Size = '{ss.HiddenSize}'");
                Logger.WriteLine($"Word Vector Size = '{ss.WordVectorSize}'");
                Logger.WriteLine($"Learning Rate = '{options.LearningRate}'");
                Logger.WriteLine($"Network Layer = '{ss.Depth}'");
                Logger.WriteLine($"Gradient Clip = '{options.GradClip}'");
                Logger.WriteLine($"Dropout Ratio = '{options.DropoutRatio}'");
                Logger.WriteLine($"Batch Size = '{options.BatchSize}'");
                Logger.WriteLine($"Arch Type = '{archType}'");

                ss.IterationDone += ss_IterationDone;
                ss.Train(100, options.LearningRate, options.GradClip);
            }
            else if (String.Equals(options.TaskName, "test", StringComparison.InvariantCultureIgnoreCase))
            {
                ss = new AttentionSeq2Seq(options.ModelFilePath, 1, archType);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(options.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    List<string> outputWords = ss.Predict(line.ToLower().Trim().Split(' ').ToList());
                    outputLines.Add(String.Join(" ", outputWords));
                }

                File.WriteAllLines(options.OutputTestFile, outputLines);
            }
            else
            {
                argParser.Usage();
            }
        }       
    }
}
