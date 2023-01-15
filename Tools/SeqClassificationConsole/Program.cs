// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using AdvUtils;
using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;

namespace SeqClassificationConsole
{
    class Program
    {
        private static SeqClassificationOptions opts = new SeqClassificationOptions();
        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (!opts.NotifyEmail.IsNullOrEmpty())
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void ShowOptions(string[] args, SeqClassificationOptions opts)
        {
            string commandLine = string.Join(" ", args);
            Logger.WriteLine($"SeqClassificationConsole v2.7.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");

            string strOpts = JsonConvert.SerializeObject( opts, Formatting.Indented, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore, Converters = new[] { new StringEnumConverter() }, } );
            Logger.WriteLine($"Configs: {strOpts}");
        }

        static void Main(string[] args)
        {
            try
            {
                //Parse command line
                //   Seq2SeqOptions opts = new Seq2SeqOptions();
                ArgParser argParser = new ArgParser(args, opts);

                if (!opts.ConfigFilePath.IsNullOrEmpty())
                {
                    Logger.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    opts = JsonConvert.DeserializeObject<SeqClassificationOptions>(File.ReadAllText(opts.ConfigFilePath));
                }

                Logger.LogFile = $"{nameof(SeqClassificationConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";
                ShowOptions(args, opts);

                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                SeqClassification ss = null;

                if ( opts.Task == ModeEnums.Train )
                {
                    // Load train corpus
                    var trainCorpus = new SeqClassificationMultiTasksCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang,  maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                        maxSentLength: opts.MaxSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence );

                    // Load valid corpus
                    var validCorpusList = new List<SeqClassificationMultiTasksCorpus>();
                    if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                    {
                        string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                        foreach (var validCorpusPath in validCorpusPathList)
                        {
                            validCorpusList.Add(new SeqClassificationMultiTasksCorpus(validCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, opts.ValMaxTokenSizePerBatch, opts.MaxSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence ));
                        }
                    }

                    // Create learning rate
                    ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

                    // Create metrics
                    Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();

                    // Create optimizer
                    IOptimizer optimizer = Misc.CreateOptimizer(opts);


                    if (!opts.ModelFilePath.IsNullOrEmpty() && File.Exists(opts.ModelFilePath))
                    {
                        //Incremental training
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new SeqClassification(opts);
                       
                        for (int i = 0; i < ss.ClsVocabs.Count; i++)
                        {
                            taskId2metrics.Add(i, new List<IMetric>());
                            taskId2metrics[i].Add(new MultiLabelsFscoreMetric("", ss.ClsVocabs[i].GetAllTokens(keepBuildInTokens: false)));
                        }
                    }
                    else
                    {
                        // Load or build vocabulary
                        Vocab srcVocab = null;
                        List<Vocab> tgtVocabs = null;
                        if (!opts.SrcVocab.IsNullOrEmpty() && !opts.TgtVocab.IsNullOrEmpty() )
                        {
                            Logger.WriteLine($"Loading source vocabulary from '{opts.SrcVocab}' and target vocabulary from '{opts.TgtVocab}'.");
                            // Vocabulary files are specified, so we load them
                            srcVocab = new Vocab(opts.SrcVocab);

                            tgtVocabs = new List<Vocab>
                            {
                                new Vocab(opts.TgtVocab)
                            };
                        }
                        else
                        {
                            Logger.WriteLine($"Building vocabulary from training corpus.");
                            // We don't specify vocabulary, so we build it from train corpus
                            (srcVocab, tgtVocabs) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize);
                        }

                        for (int i = 0; i < tgtVocabs.Count; i++)
                        {
                            taskId2metrics.Add(i, new List<IMetric>());
                            taskId2metrics[i].Add(new MultiLabelsFscoreMetric("", tgtVocabs[i].GetAllTokens(keepBuildInTokens: false)));
                        }

                        //New training
                        ss = new SeqClassification(opts, srcVocab, tgtVocabs);
                    }

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, taskId2metrics: taskId2metrics, decodingOptions: decodingOptions);
                }
                //else if (opts.Task == ModeEnums.Valid)
                //{
                //    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'");

                //    // Create metrics
                //    List<IMetric> metrics = new List<IMetric>
                //{
                //    new BleuMetric(),
                //    new LengthRatioMetric()
                //};

                //    // Load valid corpus
                //    ParallelCorpus validCorpus = new ParallelCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxSrcTestSentLength, opts.MaxTgtTestSentLength, shuffleEnums: shuffleType);

                //    ss = new Seq2Seq(opts);
                //    ss.EvaluationWatcher += ss_EvaluationWatcher;
                //    ss.Valid(validCorpus: validCorpus, metrics: metrics);
                //}
                else if ( opts.Task == ModeEnums.Test )
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new SeqClassification(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    ss.Test<SeqClassificationMultiTasksCorpusBatch>(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath);

                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                //else if (opts.Task == ModeEnums.DumpVocab)
                //{
                //    ss = new Seq2Seq(opts);
                //    ss.DumpVocabToFiles(opts.SrcVocab, opts.TgtVocab);
                //}
                else
                {
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Task '{opts.Task}' is not supported.");
                    argParser.Usage();
                }
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Exception: '{err.Message}'");
                Logger.WriteLine($"Call stack: '{err.StackTrace}'");
            }
        }       
    }
}
