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
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Applications;

namespace Seq2SeqConsole
{
    internal static class Program
    {
        private static Seq2SeqOptions opts = new Seq2SeqOptions();
        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (!opts.NotifyEmail.IsNullOrEmpty() )
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void Main(string[] args)
        {
            try
            {
                //Parse command line
                ArgParser argParser = new ArgParser(args, opts);
                if (!opts.ConfigFilePath.IsNullOrEmpty())
                {
                    Console.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    opts = JsonConvert.DeserializeObject<Seq2SeqOptions>(File.ReadAllText(opts.ConfigFilePath));                
                }

                Logger.Verbose = opts.LogVerbose;
                Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";

                ShowOptions(args, opts);

                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                Seq2Seq ss = null;
                if (opts.Task == ModeEnums.Train)
                {
                    // Load train corpus
                    var trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize,
                        maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence);

                    // Load valid corpus
                    var validCorpusList = new List<Seq2SeqCorpus>();
                    if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                    {
                        string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                        foreach (var validCorpusPath in validCorpusPathList)
                        {
                            validCorpusList.Add(new Seq2SeqCorpus(validCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence));
                        }

                    }

                    // Create learning rate
                    ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                    // Create optimizer
                    IOptimizer optimizer = Misc.CreateOptimizer(opts);

                    // Create metrics
                    List<IMetric> metrics = CreateMetrics();

                    if (!opts.ModelFilePath.IsNullOrEmpty() && File.Exists(opts.ModelFilePath))
                    {
                        //Incremental training
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new Seq2Seq(opts);
                    }
                    else
                    {
                        // Load or build vocabulary
                        Vocab srcVocab = null;
                        Vocab tgtVocab = null;
                        if (!opts.SrcVocab.IsNullOrEmpty() && !opts.TgtVocab.IsNullOrEmpty())
                        {
                            Logger.WriteLine($"Loading source vocabulary from '{opts.SrcVocab}' and target vocabulary from '{opts.TgtVocab}'. Shared vocabulary is '{opts.SharedEmbeddings}'");
                            if (opts.SharedEmbeddings == true && (opts.SrcVocab != opts.TgtVocab))
                            {
                                throw new ArgumentException("The source and target vocabularies must be identical if their embeddings are shared.");
                            }

                            // Vocabulary files are specified, so we load them
                            srcVocab = new Vocab(opts.SrcVocab);
                            tgtVocab = new Vocab(opts.TgtVocab);
                        }
                        else
                        {
                            Logger.WriteLine($"Building vocabulary from training corpus. Shared vocabulary is '{opts.SharedEmbeddings}'");
                            // We don't specify vocabulary, so we build it from train corpus

                            (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize, opts.SharedEmbeddings);
                        }

                        //New training
                        ss = new Seq2Seq(opts, srcVocab, tgtVocab);
                    }

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, metrics: metrics, decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Valid)
                {
                    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");

                    // Create metrics
                    List<IMetric> metrics = CreateMetrics();

                    // Load valid corpus
                    Seq2SeqCorpus validCorpus = new Seq2SeqCorpus(opts.ValidCorpusPaths, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence);

                    ss = new Seq2Seq(opts);
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;
                    ss.Valid(validCorpus: validCorpus, metrics: metrics, decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new Seq2Seq(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    if (String.IsNullOrEmpty(opts.OutputPromptFile))
                    {
                        ss.Test<Seq2SeqCorpusBatch>(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);
                    }
                    else
                    {
                        Logger.WriteLine($"Test with prompt file '{opts.OutputPromptFile}'");
                        ss.Test<Seq2SeqCorpusBatch>(opts.InputTestFile, opts.OutputPromptFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);
                    }

                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (opts.Task == ModeEnums.Alignment)
                {
                    if (File.Exists(opts.OutputAlignmentsFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputAlignmentsFile}' exist. Delete it.");
                        File.Delete(opts.OutputAlignmentsFile);
                    }

                    //Test trained model
                    ss = new Seq2Seq(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    if (String.IsNullOrEmpty(opts.OutputPromptFile))
                    {
                        Logger.WriteLine(Logger.Level.err, $"The prompt file for output is required for alignment task.");
                        return;
                    }

                    Logger.WriteLine($"Test with prompt file '{opts.OutputPromptFile}'");
                    ss.Test<Seq2SeqCorpusBatch>(opts.InputTestFile, opts.OutputPromptFile, null, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);

                    stopwatch.Stop();

                    Logger.WriteLine($"Alignment mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (opts.Task == ModeEnums.DumpVocab)
                {
                    ss = new Seq2Seq(opts);
                    ss.DumpVocabToFiles(opts.SrcVocab, opts.TgtVocab);
                }
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

        private static List<IMetric> CreateMetrics()
        {
            IMetric seqGenMetric = null;
            if (opts.SeqGenerationMetric.Equals("BLEU", StringComparison.InvariantCultureIgnoreCase))
            {
                seqGenMetric = new BleuMetric();
            }
            else
            {
                seqGenMetric = new RougeMetric();
            }
            List<IMetric> metrics = new List<IMetric>
                    {
                        seqGenMetric,
                        new LengthRatioMetric()
                    };
            return metrics;
        }

        private static void ShowOptions(string[] args, Seq2SeqOptions opts)
        {
            var commandLine = string.Join(" ", args);
            var strOpts = JsonConvert.SerializeObject( opts, Formatting.Indented, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore, Converters = new[] { new StringEnumConverter() }, } );

            Logger.WriteLine($"Seq2SeqSharp v2.5.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
