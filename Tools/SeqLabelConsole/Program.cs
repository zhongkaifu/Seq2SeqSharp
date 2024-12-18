﻿// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Newtonsoft.Json;
using Seq2SeqSharp;
using Seq2SeqSharp.LearningRate;
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
using Seq2SeqSharp.Enums;

namespace SeqLabelConsole
{
    internal class Program
    {
        static SeqLabelOptions opts = new SeqLabelOptions();

        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (!opts.NotifyEmail.IsNullOrEmpty())
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void Main(string[] args)
        {
            //Parse command line
            ArgParser argParser = new ArgParser(args, opts);

            Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{nameof(SeqLabelConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");
            ShowOptions(args);

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
                SeqLabelingCorpus trainCorpus = new SeqLabelingCorpus(opts.TrainCorpusPath, opts.MaxTokenSizePerBatch, maxSentLength: opts.MaxSentLength, paddingEnums: opts.PaddingType);

                // Load valid corpus
                List<SeqLabelingCorpus> validCorpusList = new List<SeqLabelingCorpus>();
                if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                {
                    string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                    foreach (var validCorpusPath in validCorpusPathList)
                    {
                        validCorpusList.Add(new SeqLabelingCorpus(validCorpusPath, opts.MaxTokenSizePerBatch, maxSentLength: opts.MaxSentLength, paddingEnums: opts.PaddingType));
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
                ILearningRate learningRate = null;
                if (opts.LearningRateType == LearningRateTypeEnums.CosineDecay)
                {
                    learningRate = new CosineDecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.LearningRateDecaySteps, opts.WeightsUpdateCount);
                }
                else
                {
                    learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);
                }

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
                sl.EvaluationWatcher += Ss_EvaluationWatcher;

                // Kick off training
                sl.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, metrics: metrics.ToArray(), decodingOptions: decodingOptions);


            }
            else if ( opts.Task == ModeEnums.Valid )
            {
                Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");

                // Load valid corpus
                SeqLabelingCorpus validCorpus = new SeqLabelingCorpus(opts.ValidCorpusPaths, opts.ValMaxTokenSizePerBatch, opts.MaxSentLength, paddingEnums: opts.PaddingType);
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
                Logger.WriteLine($"Test model '{opts.ModelFilePath}' using input data set '{opts.InputTestFile}'");

                //Test trained model
                sl = new SeqLabel(opts);
                using (StreamWriter sw = new StreamWriter(opts.OutputFile))
                {
                    foreach (string line in File.ReadLines(opts.InputTestFile))
                    {
                        var nrs = sl.Test<SeqLabelingCorpusBatch>(ConstructInputTokens(line.Trim().Split(' ').ToList()), null, decodingOptions: decodingOptions);
                        var outputLines = nrs[0].Output[0].Select(x => string.Join(" ", x));
                        foreach (var outputLine in outputLines)
                        {
                            sw.WriteLine(outputLine);
                        }
                    }
                }
            }
            else if (opts.Task == ModeEnums.DumpVocab)
            {
                sl = new SeqLabel(opts);
                sl.DumpVocabToFiles(opts.SrcVocab, opts.TgtVocab);
            }
            else
            {
                argParser.Usage();
            }
        }

        public static List<List<string>> ConstructInputTokens(List<string> input)
        {
            List<List<string>> inputSeqs = new List<List<string>>() { input };

            return inputSeqs;
        }
        private static void ShowOptions(string[] args)
        {
            string commandLine = string.Join(" ", args);
            Logger.WriteLine($"Seq2SeqSharp v2.8.16 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
        }
    }
}
