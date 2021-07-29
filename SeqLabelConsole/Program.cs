using AdvUtils;
using Newtonsoft.Json;
using Seq2SeqSharp;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SeqLabelConsole
{
    internal class Program
    {
        private static void Ss_IterationDone(object sender, EventArgs e)
        {
            CostEventArg ep = e as CostEventArg;

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

            Logger.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate:F6}, AvgCost = {ep.AvgCostInTotal:F4}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}");
        }

        private static void Main(string[] args)
        {
            ShowOptions(args);

            Logger.LogFile = $"{nameof(SeqLabelConsole)}_{Utils.GetTimeStamp(DateTime.Now)}.log";

            //Parse command line
            Options opts = new Options();
            ArgParser argParser = new ArgParser(args, opts);

            if (string.IsNullOrEmpty(opts.ConfigFilePath) == false)
            {
                Logger.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                opts = JsonConvert.DeserializeObject<Options>(File.ReadAllText(opts.ConfigFilePath));
            }


            SequenceLabel sl = null;
            ProcessorTypeEnums processorType = (ProcessorTypeEnums)Enum.Parse(typeof(ProcessorTypeEnums), opts.ProcessorType);
            EncoderTypeEnums encoderType = (EncoderTypeEnums)Enum.Parse(typeof(EncoderTypeEnums), opts.EncoderType);
            ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.TaskName);

            //Parse device ids from options          
            int[] deviceIds = opts.DeviceIds.Split(',').Select(x => int.Parse(x)).ToArray();
            if (mode == ModeEnums.Train)
            {
                // Load train corpus
                SeqLabelingCorpus trainCorpus = new SeqLabelingCorpus(opts.TrainCorpusPath, opts.BatchSize, opts.ShuffleBlockSize, maxSentLength: opts.MaxSentLength);

                // Load valid corpus
                SeqLabelingCorpus validCorpus = string.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new SeqLabelingCorpus(opts.ValidCorpusPath, opts.BatchSize, opts.ShuffleBlockSize, maxSentLength: opts.MaxSentLength);

                // Load or build vocabulary
                Vocab srcVocab = null;
                Vocab tgtVocab = null;
                if (!string.IsNullOrEmpty(opts.SrcVocab) && !string.IsNullOrEmpty(opts.TgtVocab))
                {
                    // Vocabulary files are specified, so we load them
                    srcVocab = new Vocab(opts.SrcVocab);
                    tgtVocab = new Vocab(opts.TgtVocab);
                }
                else
                {
                    // We don't specify vocabulary, so we build it from train corpus
                    (srcVocab, tgtVocab) = trainCorpus.BuildVocabs();
                }

                // Create learning rate
                ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                // Create optimizer
                IOptimizer optimizer = null;
                if (String.Equals(opts.Optimizer, "Adam", StringComparison.InvariantCultureIgnoreCase))
                {
                    optimizer = new AdamOptimizer(opts.GradClip, opts.Beta1, opts.Beta2);
                }
                else
                {
                    optimizer = new RMSPropOptimizer(opts.GradClip, opts.Beta1);
                }

                // Create metrics
                List<IMetric> metrics = new List<IMetric>();
                foreach (string word in tgtVocab.Items)
                {
                    if (BuildInTokens.IsPreDefinedToken(word) == false)
                    {
                        metrics.Add(new SequenceLabelFscoreMetric(word));
                    }
                }

                if (File.Exists(opts.ModelFilePath) == false)
                {
                    //New training
                    sl = new SequenceLabel(hiddenDim: opts.HiddenSize, embeddingDim: opts.WordVectorSize, encoderLayerDepth: opts.EncoderLayerDepth, multiHeadNum: opts.MultiHeadNum,
                        encoderType: encoderType,
                        dropoutRatio: opts.DropoutRatio, deviceIds: deviceIds, processorType: processorType, modelFilePath: opts.ModelFilePath, srcVocab: srcVocab, clsVocab: tgtVocab, maxSntSize: opts.MaxSentLength);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    sl = new SequenceLabel(modelFilePath: opts.ModelFilePath, processorType: processorType, deviceIds: deviceIds, dropoutRatio: opts.DropoutRatio, maxSntSize: opts.MaxSentLength);
                }

                // Add event handler for monitoring
                sl.StatusUpdateWatcher += Ss_IterationDone;

                // Kick off training
                sl.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, metrics: metrics);


            }
            else if (mode == ModeEnums.Valid)
            {
                Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'");

                // Load valid corpus
                SeqLabelingCorpus validCorpus = new SeqLabelingCorpus(opts.ValidCorpusPath, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);
                (Vocab srcVocab, Vocab tgtVocab) = validCorpus.BuildVocabs();

                // Create metrics
                List<IMetric> metrics = new List<IMetric>();
                foreach (string word in tgtVocab.Items)
                {
                    if (BuildInTokens.IsPreDefinedToken(word) == false)
                    {
                        metrics.Add(new SequenceLabelFscoreMetric(word));
                    }
                }

                sl = new SequenceLabel(modelFilePath: opts.ModelFilePath, processorType: processorType, deviceIds: deviceIds, maxSntSize: opts.MaxSentLength);
                sl.Valid(validCorpus: validCorpus, metrics: metrics);
            }
            else if (mode == ModeEnums.Test)
            {
                Logger.WriteLine($"Test model '{opts.ModelFilePath}' by input corpus '{opts.InputTestFile}'");

                //Test trained model
                sl = new SequenceLabel(modelFilePath: opts.ModelFilePath, processorType: processorType, deviceIds: deviceIds, maxSntSize: opts.MaxSentLength);

                List<string> outputLines = new List<string>();
                string[] data_sents_raw1 = File.ReadAllLines(opts.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    var nr = sl.Test(ConstructInputTokens(line.Trim().Split(' ').ToList()));
                    outputLines.AddRange(nr.Output[0].Select(x => string.Join(" ", x)));
                }

                File.WriteAllLines(opts.OutputTestFile, outputLines);
            }
            else
            {
                argParser.Usage();
            }
        }

        public static List<List<string>> ConstructInputTokens(List<string> input)
        {
            List<string> inputSeq = new List<string>();

            if (input != null)
            {
                inputSeq.AddRange(input);
            }

            List<List<string>> inputSeqs = new List<List<string>>() { inputSeq };

            return inputSeqs;
        }
        private static void ShowOptions(string[] args)
        {
            string commandLine = string.Join(" ", args);
            Logger.WriteLine($"Seq2SeqSharp v2.2.1 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
        }
    }
}
