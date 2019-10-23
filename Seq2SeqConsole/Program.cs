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

                Logger.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate.ToString("F6")}, Cost = {ep.CostPerWord.ToString("F4")}, AvgCost = {ep.AvgCostInTotal.ToString("F4")}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin.ToString("F")}, WordPerSec = {wordPerSec.ToString("F")}, Batch = {ep.BatchSize}");
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
            ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.TaskName);


            //Parse device ids from options          
            int[] deviceIds = opts.DeviceIds.Split(',').Select(x => int.Parse(x)).ToArray();

            if (mode == ModeEnums.Train)
            {
                ShowOptions(args, opts);

                Corpus trainCorpus = new Corpus(opts.TrainCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);
                if (File.Exists(opts.ModelFilePath) == false)
                {
                    //New training
                    ss = new AttentionSeq2Seq(embeddingDim: opts.WordVectorSize, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth,
                        trainCorpus: trainCorpus, srcVocabFilePath: opts.SrcVocab, tgtVocabFilePath: opts.TgtVocab,
                        srcEmbeddingFilePath: opts.SrcEmbeddingModelFilePath, tgtEmbeddingFilePath: opts.TgtEmbeddingModelFilePath,
                        modelFilePath: opts.ModelFilePath, batchSize: opts.BatchSize, dropoutRatio: opts.DropoutRatio,
                        archType: archType, deviceIds: deviceIds, multiHeadNum: opts.MultiHeadNum, warmupSteps: opts.WarmUpSteps, gradClip: opts.GradClip, encoderType: encoderType);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, batchSize: opts.BatchSize, archType: archType, dropoutRatio: opts.DropoutRatio, gradClip: opts.GradClip, 
                        deviceIds: deviceIds);
                    ss.TrainCorpus = trainCorpus;
                }

                ss.IterationDone += ss_IterationDone;
                ss.Train(opts.MaxEpochNum, opts.LearningRate);
            }
            else if (mode == ModeEnums.Test)
            {
                //Test trained model
                ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, batchSize: 1, archType: archType, dropoutRatio: 0.0f, gradClip: 0.0f, deviceIds: deviceIds);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(opts.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    List<List<string>> outputWordsList = ss.Predict(line.ToLower().Trim().Split(' ').ToList(), opts.BeamSearch);
                    outputLines.AddRange(outputWordsList.Select(x => String.Join(" ", x)));
                }

                File.WriteAllLines(opts.OutputTestFile, outputLines);
            }
            else if (mode == ModeEnums.VisualizeNetwork)
            {
                ss = new AttentionSeq2Seq(embeddingDim: opts.WordVectorSize, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, 
                    decoderLayerDepth: opts.DecoderLayerDepth,trainCorpus: null, srcVocabFilePath: null, tgtVocabFilePath: null,
                    srcEmbeddingFilePath: null, tgtEmbeddingFilePath: null,
                    modelFilePath: opts.ModelFilePath, batchSize: 1, dropoutRatio: opts.DropoutRatio,
                    archType: archType, deviceIds: new int[1] { 0 }, multiHeadNum: opts.MultiHeadNum, 
                    warmupSteps: opts.WarmUpSteps, gradClip: opts.GradClip, encoderType: encoderType);

                ss.VisualizeNeuralNetwork(opts.VisualizeNNFilePath);
            }
            else
            {
                argParser.Usage();
            }
        }

        private static void ShowOptions(string[] args, Options options)
        {
            string commandLine = String.Join(" ", args);
            Logger.WriteLine($"Seq2SeqSharp v2.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");

            Logger.WriteLine($"Source Language = '{options.SrcLang}'");
            Logger.WriteLine($"Target Language = '{options.TgtLang}'");
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
            Logger.WriteLine($"Maxmium Sentence Length = '{options.MaxSentLength}'");
            Logger.WriteLine($"Maxmium Epoch Number = '{options.MaxEpochNum}'");
            Logger.WriteLine($"Warming Up Steps = '{options.WarmUpSteps}'");
        }
    }
}
