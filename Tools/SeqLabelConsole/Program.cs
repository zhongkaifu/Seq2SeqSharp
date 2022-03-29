using AdvUtils;
using Newtonsoft.Json;
using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
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
        private static void Main(string[] args)
        {
            ShowOptions(args);

            Logger.LogFile = $"{nameof(SeqLabelConsole)}_{Utils.GetTimeStamp(DateTime.Now)}.log";

            //Parse command line
            SeqLabelOptions opts = new SeqLabelOptions();
            ArgParser argParser = new ArgParser(args, opts);

            if (!opts.ConfigFilePath.IsNullOrEmpty())
            {
                Logger.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                opts = JsonConvert.DeserializeObject<SeqLabelOptions>(File.ReadAllText(opts.ConfigFilePath));
            }

            DecodingOptions decodingOptions = opts.CreateDecodingOptions();

            SeqLabel sl = null;

            //Parse device ids from options          
            int[] deviceIds = opts.DeviceIds.Split(',').Select(x => int.Parse(x)).ToArray();
            if ( opts.Task == ModeEnums.Train )
            {
                // Load train corpus
                SeqLabelingCorpus trainCorpus = new SeqLabelingCorpus(opts.TrainCorpusPath, opts.BatchSize, opts.ShuffleBlockSize, maxSentLength: opts.MaxTrainSentLength);

                // Load valid corpus
                List<SeqLabelingCorpus> validCorpusList = new List<SeqLabelingCorpus>();
                if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                {
                    string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                    foreach (var validCorpusPath in validCorpusPathList)
                    {
                        validCorpusList.Add(new SeqLabelingCorpus(opts.ValidCorpusPaths, opts.BatchSize, opts.ShuffleBlockSize, maxSentLength: opts.MaxTestSentLength));
                    }
                }

                // Load or build vocabulary
                Vocab srcVocab = null;
                Vocab tgtVocab = null;
                if (!opts.SrcVocab.IsNullOrEmpty() && !opts.TgtVocab.IsNullOrEmpty() )
                {
                    // Vocabulary files are specified, so we load them
                    srcVocab = new Vocab(opts.SrcVocab);
                    tgtVocab = new Vocab(opts.TgtVocab);
                }
                else
                {
                    // We don't specify vocabulary, so we build it from train corpus
                    (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize);
                }

                // Create learning rate
                ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                // Create optimizer
                IOptimizer optimizer = Misc.CreateOptimizer(opts);

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
                    sl = new SeqLabel(opts, srcVocab: srcVocab, clsVocab: tgtVocab);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    sl = new SeqLabel(opts);
                }

                // Add event handler for monitoring
                sl.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;

                // Kick off training
                sl.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, metrics: metrics, decodingOptions: decodingOptions);


            }
            else if ( opts.Task == ModeEnums.Valid )
            {
                Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");

                // Load valid corpus
                SeqLabelingCorpus validCorpus = new SeqLabelingCorpus(opts.ValidCorpusPaths, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxTestSentLength);
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

                sl = new SeqLabel(opts);
                sl.Valid(validCorpus: validCorpus, metrics: metrics, decodingOptions: decodingOptions);
            }
            else if ( opts.Task == ModeEnums.Test )
            {
                Logger.WriteLine($"Test model '{opts.ModelFilePath}' by input corpus '{opts.InputTestFile}'");

                //Test trained model
                sl = new SeqLabel(opts);

                List<string> outputLines = new List<string>();
                string[] data_sents_raw1 = File.ReadAllLines(opts.InputTestFile);
                foreach (string line in data_sents_raw1)
                {
                    var nrs = sl.Test<SeqLabelingCorpusBatch>(ConstructInputTokens(line.Trim().Split(' ').ToList()), null, decodingOptions: decodingOptions);
                    outputLines.AddRange(nrs[0].Output[0].Select(x => string.Join(" ", x)));
                }

                File.WriteAllLines(opts.OutputFile, outputLines);
            }
            else
            {
                argParser.Usage();
            }
        }

        public static List<List<List<string>>> ConstructInputTokens(List<string> input)
        {
            List<List<string>> inputSeqs = new List<List<string>>() { input };
            List<List<List<string>>> inputSeqsGroups = new List<List<List<string>>>() { inputSeqs };

            return inputSeqsGroups;
        }
        private static void ShowOptions(string[] args)
        {
            string commandLine = string.Join(" ", args);
            Logger.WriteLine($"Seq2SeqSharp v2.2.1 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
        }
    }
}
