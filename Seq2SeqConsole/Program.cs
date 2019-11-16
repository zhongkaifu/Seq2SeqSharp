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
using Seq2SeqSharp.Metrics;
using Newtonsoft.Json;

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

                Logger.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate.ToString("F6")}, Cost = {ep.CostPerWord.ToString("F4")}, AvgCost = {ep.AvgCostInTotal.ToString("F4")}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin.ToString("F")}, WordPerSec = {wordPerSec.ToString("F")}");
            }

        }

        public static String GetTimeStamp(DateTime timeStamp)
        {
            return String.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }

        static void Main(string[] args)
        {
            Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{GetTimeStamp(DateTime.Now)}.log";
            ShowOptions(args);

            //Parse command line
            Options opts = new Options();
            ArgParser argParser = new ArgParser(args, opts);

            if (String.IsNullOrEmpty(opts.ConfigFilePath) == false)
            {
                Logger.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                opts = JsonConvert.DeserializeObject<Options>(File.ReadAllText(opts.ConfigFilePath));
            }

            AttentionSeq2Seq ss = null;
            ProcessorTypeEnums processorType = (ProcessorTypeEnums)Enum.Parse(typeof(ProcessorTypeEnums), opts.ProcessorType);
            EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), opts.EncoderType);
            ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.TaskName);

            //Parse device ids from options          
            int[] deviceIds = opts.DeviceIds.Split(',').Select(x => int.Parse(x)).ToArray();
            if (mode == ModeEnums.Train)
            {
                // Load train corpus
                ParallelCorpus trainCorpus = new ParallelCorpus(opts.TrainCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);
                // Load valid corpus
                ParallelCorpus validCorpus = String.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new ParallelCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);

                // Load or build vocabulary
                Vocab vocab = null;
                if (!String.IsNullOrEmpty(opts.SrcVocab) && !String.IsNullOrEmpty(opts.TgtVocab))
                {
                    // Vocabulary files are specified, so we load them
                    vocab = new Vocab(opts.SrcVocab, opts.TgtVocab);
                }
                else
                {
                    // We don't specify vocabulary, so we build it from train corpus
                    vocab = new Vocab(trainCorpus);
                }

                // Create learning rate
                ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                // Create optimizer
                Optimizer optimizer = new Optimizer(opts.GradClip);

                // Create metrics
                List<IMetric> metrics = new List<IMetric>();
                metrics.Add(new BleuMetric());
                metrics.Add(new LengthRatioMetric());

                if (File.Exists(opts.ModelFilePath) == false)
                {
                    //New training
                    ss = new AttentionSeq2Seq(embeddingDim: opts.WordVectorSize, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth,
                        srcEmbeddingFilePath: opts.SrcEmbeddingModelFilePath, tgtEmbeddingFilePath: opts.TgtEmbeddingModelFilePath, vocab: vocab, modelFilePath: opts.ModelFilePath, 
                        dropoutRatio: opts.DropoutRatio, processorType: processorType, deviceIds: deviceIds, multiHeadNum: opts.MultiHeadNum, encoderType: encoderType);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, processorType: processorType, dropoutRatio: opts.DropoutRatio, deviceIds: deviceIds);
                }

                // Add event handler for monitoring
                ss.IterationDone += ss_IterationDone;

                // Kick off training
                ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, metrics: metrics);
            }
            else if (mode == ModeEnums.Valid)
            {
                Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'");

                // Create metrics
                List<IMetric> metrics = new List<IMetric>();
                metrics.Add(new BleuMetric());
                metrics.Add(new LengthRatioMetric());

                // Load valid corpus
                ParallelCorpus validCorpus = new ParallelCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);

                ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, processorType: processorType, deviceIds: deviceIds);
                ss.Valid(validCorpus: validCorpus, metrics: metrics);
            }
            else if (mode == ModeEnums.Test)
            {
                Logger.WriteLine($"Test model '{opts.ModelFilePath}' by input corpus '{opts.InputTestFile}'");

                //Test trained model
                ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, processorType: processorType, deviceIds: deviceIds);

                List<string> outputLines = new List<string>();
                var data_sents_raw1 = File.ReadAllLines(opts.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    //// Below support beam search
                    //List<List<string>> outputWordsList = ss.Predict(line.ToLower().Trim().Split(' ').ToList(), opts.BeamSearch);
                    //outputLines.AddRange(outputWordsList.Select(x => String.Join(" ", x)));

                    var outputTokensBatch = ss.Test(ParallelCorpus.ConstructInputTokens(line.ToLower().Trim().Split(' ').ToList()));
                    outputLines.AddRange(outputTokensBatch.Select(x => String.Join(" ", x)));
                }

                File.WriteAllLines(opts.OutputTestFile, outputLines);
            }
            else if (mode == ModeEnums.VisualizeNetwork)
            {
                ss = new AttentionSeq2Seq(embeddingDim: opts.WordVectorSize, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth, 
                    vocab: new Vocab(), srcEmbeddingFilePath: null, tgtEmbeddingFilePath: null, modelFilePath: opts.ModelFilePath, dropoutRatio: opts.DropoutRatio,
                    processorType: processorType, deviceIds: new int[1] { 0 }, multiHeadNum: opts.MultiHeadNum, encoderType: encoderType);

                ss.VisualizeNeuralNetwork(opts.VisualizeNNFilePath);
            }
            else
            {
                argParser.Usage();
            }
        }

        private static void ShowOptions(string[] args)
        {
            string commandLine = String.Join(" ", args);
            Logger.WriteLine($"Seq2SeqSharp v2.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");      
        }
    }
}
