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

                Logger.WriteLine($"Update = '{ep.Update}' Epoch = '{ep.Epoch}' LR = '{ep.AvgLearningRate.ToString("F6")}', Current Cost = '{ep.CostPerWord.ToString("F6")}', Avg Cost = '{ep.avgCostInTotal.ToString("F6")}', SentInTotal = '{ep.ProcessedSentencesInTotal}', SentPerMin = '{sentPerMin.ToString("F")}', WordPerSec = '{wordPerSec.ToString("F")}'");
            }

        }

        public static String GetTimeStamp(DateTime timeStamp)
        {
            return String.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }

        static void Main(string[] args)
        {
            Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{GetTimeStamp(DateTime.Now)}.log";

            //Parse command line
            Options opts = new Options();
            ArgParser argParser = new ArgParser(args, opts);

            AttentionSeq2Seq ss = null;
            ArchTypeEnums archType = (ArchTypeEnums)Enum.Parse(typeof(ArchTypeEnums), opts.ArchType);
            EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), opts.EncoderType);

            //Parse device ids from options          
            int[] deviceIds = opts.DeviceIds.Split(',').Select(x => int.Parse(x)).ToArray();

            if (String.Equals(opts.TaskName, "train", StringComparison.InvariantCultureIgnoreCase))
            {
                ShowOptions(args, opts);

                Corpus trainCorpus = new Corpus(opts.TrainCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize * deviceIds.Length, 
                    opts.ShuffleBlockSize, opts.MaxSentLength);
                if (File.Exists(opts.ModelFilePath) == false)
                {
                    //New training
                    ss = new AttentionSeq2Seq(inputSize: opts.WordVectorSize, hiddenSize: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth, 
                        trainCorpus: trainCorpus, srcVocabFilePath: opts.SrcVocab, tgtVocabFilePath: opts.TgtVocab,
                        srcEmbeddingFilePath: opts.SrcEmbeddingModelFilePath, tgtEmbeddingFilePath: opts.TgtEmbeddingModelFilePath,
                        modelFilePath: opts.ModelFilePath, batchSize: opts.BatchSize, dropoutRatio: opts.DropoutRatio,
                        archType: archType, deviceIds: deviceIds, multiHeadNum: opts.MultiHeadNum, encoderType: encoderType);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(opts.ModelFilePath, opts.BatchSize, archType, deviceIds);
                    ss.TrainCorpus = trainCorpus;
                }

                ss.IterationDone += ss_IterationDone;
                ss.Train(opts.MaxEpochNum, opts.LearningRate, opts.GradClip);
            }
            else if (String.Equals(opts.TaskName, "test", StringComparison.InvariantCultureIgnoreCase))
            {
                //Test trained model
                ss = new AttentionSeq2Seq(opts.ModelFilePath, 1, archType, deviceIds);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(opts.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    List<List<string>> outputWordsList = ss.Predict(line.ToLower().Trim().Split(' ').ToList(), opts.BeamSearch);
                    outputLines.AddRange(outputWordsList.Select(x => String.Join(" ", x)));
                }

                File.WriteAllLines(opts.OutputTestFile, outputLines);
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

            Logger.WriteLine($"Source Language = '{options.SrcLang}'");
            Logger.WriteLine($"Target Language = '{options.TgtLang}'");
            Logger.WriteLine($"SSE Enable = '{System.Numerics.Vector.IsHardwareAccelerated}'");
            Logger.WriteLine($"SSE Size = '{System.Numerics.Vector<float>.Count * 32}'");
            Logger.WriteLine($"Processor counter = '{Environment.ProcessorCount}'");
            Logger.WriteLine($"Hidden Size = '{options.HiddenSize}'");
            Logger.WriteLine($"Word Vector Size = '{options.WordVectorSize}'");
            Logger.WriteLine($"Learning Rate = '{options.LearningRate}'");
            Logger.WriteLine($"Encoder Layer Depth = '{options.EncoderLayerDepth}'");
            Logger.WriteLine($"Decoder Layer Depth = '{options.DecoderLayerDepth}'");
            Logger.WriteLine($"Gradient Clip = '{options.GradClip}'");
            Logger.WriteLine($"Dropout Ratio = '{options.DropoutRatio}'");
            Logger.WriteLine($"Batch Size = '{options.BatchSize}'");
            Logger.WriteLine($"Arch Type = '{options.ArchType}'");
            Logger.WriteLine($"Encoder Type = '{options.EncoderType}'");
            Logger.WriteLine($"Device Ids = '{options.DeviceIds}'");
            Logger.WriteLine($"Maxmium Epoch Number = '{options.MaxEpochNum}'");
            Logger.WriteLine($"Maxmium Sentence Length = '{options.MaxSentLength}'");
        }
    }
}
