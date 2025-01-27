// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using AdvUtils;
using Seq2SeqSharp;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Applications;

namespace GPTConsole
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


        private static void Ss_KVCacheRemoveWatcher(object sender, EventArgs e)
        {
            KVCacheRemoveEventArg ep = e as KVCacheRemoveEventArg;
            if (ep.Reason != "Removed")
            {
                Logger.WriteLine(Logger.Level.debug, $"KV Cache Removed due to '{ep.Reason}' Key = '{ep.Key}'");
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
                    argParser = new ArgParser(args, opts);
                }

                Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{nameof(GPTConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");

                if ((opts.LogLevel & Logger.Level.debug) == Logger.Level.debug)
                {
                    ShowOptions(args, opts);
                }

                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                GPT ss = null;
                if (opts.Task == ModeEnums.Train)
                {
                    // Load train corpus
                    var trainCorpus = new SeqCorpus(corpusFilePath: opts.TrainCorpusPath, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                        maxTgtSentLength: opts.MaxTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence, indexedFilePath: opts.IndexedCorpusPath, startBatchId: opts.StartBatchId, dataPassword: opts.DataPassword);

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

                    if (!opts.ModelFilePath.IsNullOrEmpty() && File.Exists(opts.ModelFilePath))
                    {
                        //Incremental training
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new GPT(opts);
                    }
                    else
                    {
                        // Load or build vocabulary
                        Vocab tgtVocab = null;
                        if (!opts.TgtVocab.IsNullOrEmpty())
                        {
                            Logger.WriteLine($"Loading target vocabulary from '{opts.TgtVocab}'");

                            // Vocabulary files are specified, so we load them
                            tgtVocab = new Vocab(opts.TgtVocab);
                        }
                        else
                        {
                            Logger.WriteLine($"Building vocabulary from training corpus.");
                            // We don't specify vocabulary, so we build it from train corpus

                            tgtVocab = trainCorpus.BuildVocabs(opts.TgtVocabSize, opts.MinTokenFreqInVocab);

                            Logger.WriteLine($"Dump vocabulary to file '{opts.ModelFilePath}.vocab'");
                            tgtVocab.DumpVocab(opts.ModelFilePath + ".vocab");
                        }

                        //New training
                        ss = new GPT(opts, tgtVocab);
                    }

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: null, learningRate: learningRate, optimizer: optimizer, metrics: null, decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.DPO)
                {
                    Logger.WriteLine($"Starting to run DPO against model '{opts.ModelFilePath}'");


                    if (opts.ModelFilePath.IsNullOrEmpty() || !File.Exists(opts.ModelFilePath))
                    {
                        Logger.WriteLine(Logger.Level.err, $"Model '{opts.ModelFilePath}' doesn't exist.");
                        return;
                    }
                    // Load train corpus
                    var trainCorpus = new DPOCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                    maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence, indexedFilePath: opts.IndexedCorpusPath,
                    startBatchId: opts.StartBatchId, dataPassword: opts.DataPassword);

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

                    DPO trainer = new DPO(opts);

                    // Add event handler for monitoring
                    trainer.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcherDPO;
                    trainer.EvaluationWatcher += Ss_EvaluationWatcher;

                    trainer.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: null, learningRate: learningRate, optimizer: optimizer, metrics: null, decodingOptions: decodingOptions);
                }
                else if (opts.Task == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new GPT(opts);
                    ss.KVCacheRemoveWatcher += Ss_KVCacheRemoveWatcher;
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    if (String.IsNullOrEmpty(opts.OutputPromptFile))
                    {
                        ss.Test(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);
                    }
                    else
                    {
                        Logger.WriteLine($"Test with prompt file '{opts.OutputPromptFile}'");
                        ss.Test(opts.InputTestFile, opts.OutputPromptFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, opts.OutputAlignmentsFile);
                    }

                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (opts.Task == ModeEnums.DumpVocab)
                {
                    ss = new GPT(opts);
                    ss.DumpVocabToFiles(opts.TgtVocab);
                }
                else if (opts.Task == ModeEnums.UpdateVocab)
                {
                    ss = new GPT(opts);
                    Vocab tgtVocab = null;
                    if (String.IsNullOrEmpty(opts.TgtVocab) == false)
                    {
                        Logger.WriteLine($"Replacing target vocabulary in model '{opts.ModelFilePath}' by external vocabulary '{opts.TgtVocab}'");
                        tgtVocab = new Vocab(opts.TgtVocab);
                    }

                    ss.UpdateVocabs(tgtVocab);
                }
                else if (opts.Task == ModeEnums.VQModel)
                {
                    Logger.WriteLine($"Model vector quantization for '{opts.ModelFilePath}'. Type = '{opts.VQType}'");
                    ss = new GPT(opts);
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

        private static void ShowOptions(string[] args, Seq2SeqOptions opts)
        {
            var commandLine = string.Join(" ", args);
            var strOpts = JsonConvert.SerializeObject(opts, Newtonsoft.Json.Formatting.Indented, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore, Converters = new[] { new StringEnumConverter() }, });

            Logger.WriteLine($"Seq2SeqSharp v2.8.16 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
