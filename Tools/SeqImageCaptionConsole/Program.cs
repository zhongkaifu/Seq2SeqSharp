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

namespace SeqImageCaptionConsole
{
    internal static class Program
    {
        private static ImageCaptionOptions opts = new ImageCaptionOptions();

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
                        opts = JsonConvert.DeserializeObject<ImageCaptionOptions>(File.ReadAllText(opts.ConfigFilePath));
                        argParser = new ArgParser(args, opts);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to parse config file. Error = '{ex.Message}', Stack = '{ex.StackTrace}'");
                        return;
                    }
                }

                Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{nameof(SeqImageCaptionConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");
                ShowOptions(args, opts);

                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                Seq2Seq ss = null;

                if (opts.Task == ModeEnums.Train)
                {
                    var trainCorpus = new VisionTextCorpus<VisionTextCorpusBatch>(opts.TrainCorpusPath, opts.SrcLang, opts.TgtLang, opts.MaxTokenSizePerBatch,
                        opts.MaxSrcSentLength, opts.MaxTgtSentLength, opts.PaddingType, opts.TooLongSequence, opts.IndexedCorpusPath);

                    var validCorpusList = new List<ICorpus<IPairBatch>>();
                    if (!opts.ValidCorpusPaths.IsNullOrEmpty())
                    {
                        foreach (var validCorpusPath in opts.ValidCorpusPaths.Split(';', StringSplitOptions.RemoveEmptyEntries))
                        {
                            validCorpusList.Add(new VisionTextCorpus<VisionTextCorpusBatch>(validCorpusPath.Trim(), opts.SrcLang, opts.TgtLang,
                                opts.ValMaxTokenSizePerBatch, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, opts.PaddingType, opts.TooLongSequence));
                        }
                    }

                    ILearningRate learningRate = opts.LearningRateType == LearningRateTypeEnums.CosineDecay
                        ? new CosineDecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.LearningRateDecaySteps, opts.WeightsUpdateCount)
                        : new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

                    IOptimizer optimizer = Misc.CreateOptimizer(opts);
                    List<IMetric> metrics = CreateMetrics();

                    if (!opts.ModelFilePath.IsNullOrEmpty() && File.Exists(opts.ModelFilePath))
                    {
                        ss = new Seq2Seq(opts);
                    }
                    else
                    {
                        Vocab tgtVocab;
                        if (!opts.TgtVocab.IsNullOrEmpty())
                        {
                            tgtVocab = new Vocab(opts.TgtVocab);
                        }
                        else
                        {
                            tgtVocab = trainCorpus.BuildVocabs(opts.TgtVocabSize, opts.MinTokenFreqInVocab);
                            tgtVocab.DumpVocab(opts.ModelFilePath + ".tgt_vocab");
                        }

                        ss = new Seq2Seq(opts, null, tgtVocab);
                    }

                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate,
                        optimizer: optimizer, metrics: metrics.ToArray(), decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Valid)
                {
                    var validCorpus = new VisionTextCorpus<VisionTextCorpusBatch>(opts.ValidCorpusPaths, opts.SrcLang, opts.TgtLang,
                        opts.ValMaxTokenSizePerBatch, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, opts.PaddingType, opts.TooLongSequence);

                    ss = new Seq2Seq(opts);
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;
                    ss.ValidVision(validCorpus, CreateMetrics(), decodingOptions);
                }
                else if (opts.Task == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exists. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    ss = new Seq2Seq(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    ss.TestVision(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);
                    stopwatch.Stop();
                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (opts.Task == ModeEnums.DumpVocab)
                {
                    ss = new Seq2Seq(opts);
                    ss.DumpVocabToFiles(opts.ModelFilePath + ".src_vocab", opts.ModelFilePath + ".tgt_vocab");
                }
                else if (opts.Task == ModeEnums.UpdateVocab)
                {
                    if (opts.TgtVocab.IsNullOrEmpty())
                    {
                        throw new ArgumentException("--TgtVocab must be provided when updating vocabularies.");
                    }

                    ss = new Seq2Seq(opts);
                    ss.UpdateVocabs(null, new Vocab(opts.TgtVocab));
                }
                else if (opts.Task == ModeEnums.Help)
                {
                    argParser.Usage();
                }
                else
                {
                    throw new NotSupportedException($"Task '{opts.Task}' is not supported in SeqImageCaptionConsole.");
                }
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Exception: '{err.Message}'");
                Logger.WriteLine($"Call stack: '{err.StackTrace}'");
                throw;
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

        private static void ShowOptions(string[] args, ImageCaptionOptions options)
        {
            var commandLine = string.Join(" ", args);
            var strOpts = JsonConvert.SerializeObject(options, Formatting.Indented, new JsonSerializerSettings
            {
                NullValueHandling = NullValueHandling.Ignore,
                Converters = new[] { new StringEnumConverter() },
            });

            Logger.WriteLine($"SeqImageCaptionConsole based on Seq2SeqSharp v2.8.21");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
