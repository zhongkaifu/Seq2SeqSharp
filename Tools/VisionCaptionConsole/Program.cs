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
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace VisionCaptionConsole
{
    internal static class Program
    {
        private static Seq2SeqOptions opts = new Seq2SeqOptions();

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
            try
            {
                ArgParser argParser = new ArgParser(args, opts);
                if (!opts.ConfigFilePath.IsNullOrEmpty())
                {
                    Console.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    try
                    {
                        opts = JsonConvert.DeserializeObject<Seq2SeqOptions>(File.ReadAllText(opts.ConfigFilePath));
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to parse config file. Error = '{ex.Message}', Stack = '{ex.StackTrace}'");
                        return;
                    }
                }

                Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{nameof(VisionCaptionConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");

                ShowOptions(args, opts);

                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                VisionCaption ss = null;

                if (opts.Task == ModeEnums.Train)
                {
                    var trainCorpus = new VisionTextCorpus<VisionTextCorpusBatch>(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang,
                        maxTokenSizePerBatch: opts.MaxTokenSizePerBatch, maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength,
                        paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence, indexedFilePath: opts.IndexedCorpusPath);

                    var validCorpusList = new List<VisionTextCorpus<VisionTextCorpusBatch>>();
                    if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                    {
                        string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                        foreach (var validCorpusPath in validCorpusPathList)
                        {
                            validCorpusList.Add(new VisionTextCorpus<VisionTextCorpusBatch>(validCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValMaxTokenSizePerBatch,
                                opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence));
                        }
                    }

                    ILearningRate learningRate = opts.LearningRateType == LearningRateTypeEnums.CosineDecay
                        ? new CosineDecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.LearningRateDecaySteps, opts.WeightsUpdateCount)
                        : new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

                    IOptimizer optimizer = Misc.CreateOptimizer(opts);
                    List<IMetric> metrics = CreateMetrics();

                    if (!opts.ModelFilePath.IsNullOrEmpty() && File.Exists(opts.ModelFilePath))
                    {
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new VisionCaption(opts);
                    }
                    else
                    {
                        Vocab tgtVocab;
                        if (!opts.TgtVocab.IsNullOrEmpty())
                        {
                            Logger.WriteLine($"Loading target vocabulary from '{opts.TgtVocab}'.");
                            tgtVocab = new Vocab(opts.TgtVocab);
                        }
                        else
                        {
                            Logger.WriteLine("Building target vocabulary from training corpus.");
                            tgtVocab = trainCorpus.BuildVocabs(opts.TgtVocabSize, opts.MinTokenFreqInVocab);
                            Logger.WriteLine($"Dump target vocabulary to file '{opts.ModelFilePath}.tgt_vocab'");
                            tgtVocab.DumpVocab(opts.ModelFilePath + ".tgt_vocab");
                        }

                        ss = new VisionCaption(opts, tgtVocab);
                    }

                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate,
                        optimizer: optimizer, metrics: metrics.ToArray(), decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Valid)
                {
                    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");
                    List<IMetric> metrics = CreateMetrics();
                    var validCorpus = new VisionTextCorpus<VisionTextCorpusBatch>(opts.ValidCorpusPaths, opts.SrcLang, opts.TgtLang, opts.ValMaxTokenSizePerBatch,
                        opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence);

                    ss = new VisionCaption(opts);
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;
                    ss.Valid(validCorpus: validCorpus, metrics: metrics, decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exists. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    if (!opts.OutputPromptFile.IsNullOrEmpty())
                    {
                        Logger.WriteLine(Logger.Level.warn, "Prompt files are ignored in image caption mode.");
                    }

                    ss = new VisionCaption(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    ss.Test(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions);
                    stopwatch.Stop();
                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (opts.Task == ModeEnums.Alignment)
                {
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, "Alignment mode is not supported for image captioning.");
                    argParser.Usage();
                }
                else if (opts.Task == ModeEnums.DumpVocab)
                {
                    ss = new VisionCaption(opts);
                    ss.DumpVocabToFiles(opts.SrcVocab, opts.TgtVocab);
                }
                else if (opts.Task == ModeEnums.UpdateVocab)
                {
                    ss = new VisionCaption(opts);
                    Vocab tgtVocab = null;
                    if (!opts.TgtVocab.IsNullOrEmpty())
                    {
                        Logger.WriteLine($"Replacing target vocabulary in model '{opts.ModelFilePath}' with '{opts.TgtVocab}'");
                        tgtVocab = new Vocab(opts.TgtVocab);
                    }
                    ss.UpdateVocabs(null, tgtVocab);
                }
                else if (opts.Task == ModeEnums.VQModel)
                {
                    Logger.WriteLine($"Model vector quantization for '{opts.ModelFilePath}'. Type = '{opts.VQType}'");
                    ss = new VisionCaption(opts);
                    ss.VQModel();
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
            IMetric seqGenMetric = opts.SeqGenerationMetric.Equals("BLEU", StringComparison.InvariantCultureIgnoreCase)
                ? new BleuMetric()
                : new RougeMetric();

            return new List<IMetric>
            {
                seqGenMetric,
                new LengthRatioMetric()
            };
        }

        private static void ShowOptions(string[] args, Seq2SeqOptions opts)
        {
            var commandLine = string.Join(" ", args);
            var strOpts = JsonConvert.SerializeObject(opts, Formatting.Indented, new JsonSerializerSettings()
            {
                NullValueHandling = NullValueHandling.Ignore,
                Converters = new[] { new StringEnumConverter() },
            });

            Logger.WriteLine($"Seq2SeqSharp v2.8.16 Vision Caption Console");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
