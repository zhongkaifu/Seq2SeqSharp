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

namespace Seq2SeqClassificationConsole
{
    internal class Program
    {
        private static Seq2SeqClassificationOptions opts = new Seq2SeqClassificationOptions();
        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if ( !opts.NotifyEmail.IsNullOrEmpty() )
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

                if ( !opts.ConfigFilePath.IsNullOrEmpty() )
                {
                    Console.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    opts = JsonConvert.DeserializeObject<Seq2SeqClassificationOptions>(File.ReadAllText(opts.ConfigFilePath));
                }

                Logger.LogFile = $"{nameof(Seq2SeqClassificationConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";
                ShowOptions(args, opts);

                Seq2SeqClassification ss = null;

                if ( opts.Task == ModeEnums.Train )
                {
                    // Load train corpus
                    var trainCorpus = new Seq2SeqClassificationCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize, shuffleBlockSize: opts.ShuffleBlockSize,
                        maxSrcSentLength: opts.MaxTrainSrcSentLength, maxTgtSentLength: opts.MaxTrainTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence );

                    // Load valid corpus
                    var validCorpusList = new List<Seq2SeqClassificationCorpus>();
                    if (!opts.ValidCorpusPaths.IsNullOrEmpty() )
                    {
                        string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
                        foreach (var validCorpusPath in validCorpusPathList)
                        {
                            validCorpusList.Add(new Seq2SeqClassificationCorpus(validCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence ));
                        }
                    }

                    // Create learning rate
                    ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                    // Create optimizer
                    IOptimizer optimizer = Misc.CreateOptimizer(opts);

                    // Create metrics
                    IMetric seqGenMetric = null;
                    if (opts.SeqGenerationMetric.Equals("BLEU", StringComparison.InvariantCultureIgnoreCase))
                    {
                        seqGenMetric = new BleuMetric();
                    }
                    else
                    {
                        seqGenMetric = new RougeMetric();
                    }

                    Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
                    List<IMetric> task1Metrics = new List<IMetric>
                    {
                        seqGenMetric,
                        new LengthRatioMetric()
                    };

                    taskId2metrics.Add(1, task1Metrics);

                    var task0Metrics = new List<IMetric>();
                    if ( !opts.ModelFilePath.IsNullOrEmpty() && File.Exists( opts.ModelFilePath ) )
                    {
                        //Incremental training
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new Seq2SeqClassification(opts);
                        task0Metrics.Add(new MultiLabelsFscoreMetric("", ss.ClsVocab.GetAllTokens(keepBuildInTokens: false)));
                    }
                    else
                    {
                        // Load or build vocabulary
                        Vocab srcVocab = null;
                        Vocab tgtVocab = null;
                        Vocab clsVocab = null;
                        if (!opts.SrcVocab.IsNullOrEmpty() && !opts.TgtVocab.IsNullOrEmpty() && !opts.ClsVocab.IsNullOrEmpty() )
                        {
                            Logger.WriteLine($"Loading source vocabulary from '{opts.SrcVocab}' and target vocabulary from '{opts.TgtVocab}' and classification vocabulary from '{opts.ClsVocab}'. Shared vocabulary is '{opts.SharedEmbeddings}'");
                            if (opts.SharedEmbeddings == true && (opts.SrcVocab != opts.TgtVocab))
                            {
                                throw new ArgumentException("The source and target vocabularies must be identical if their embeddings are shared.");
                            }

                            // Vocabulary files are specified, so we load them
                            srcVocab = new Vocab(opts.SrcVocab);
                            tgtVocab = new Vocab(opts.TgtVocab);
                            clsVocab = new Vocab(opts.ClsVocab);
                        }
                        else
                        {
                            Logger.WriteLine($"Building vocabulary from training corpus. Shared vocabulary is '{opts.SharedEmbeddings}'");

                            if (!opts.SrcVocab.IsNullOrEmpty() )
                            {
                                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Source vocabulary '{opts.SrcVocab}' is not empty, but we will build it from training corpus.");
                            }

                            if (!opts.TgtVocab.IsNullOrEmpty() )
                            {
                                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Target vocabulary '{opts.TgtVocab}' is not empty, but we will build it from training corpus.");
                            }

                            if (!opts.ClsVocab.IsNullOrEmpty() )
                            {
                                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Classification vocabulary '{opts.ClsVocab}' is not empty, but we will build it from training corpus.");
                            }


                            // We don't specify vocabulary, so we build it from train corpus
                            (srcVocab, tgtVocab, clsVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize, opts.SharedEmbeddings);
                        }

                        //New training
                        ss = new Seq2SeqClassification(opts, srcVocab, tgtVocab, clsVocab);
                        task0Metrics.Add(new MultiLabelsFscoreMetric("", clsVocab.GetAllTokens(keepBuildInTokens: false)));
                    }

                    taskId2metrics.Add(0, task0Metrics);

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, taskId2metrics: taskId2metrics);
                }
                else if ( opts.Task == ModeEnums.Valid )
                {
                    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");

                    // Load valid corpus
                    Seq2SeqClassificationCorpus validCorpus = new Seq2SeqClassificationCorpus(opts.ValidCorpusPaths, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence );

                    ss = new Seq2SeqClassification(opts);
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Create metrics
                    IMetric seqGenMetric = null;
                    if (opts.SeqGenerationMetric.Equals("BLEU", StringComparison.InvariantCultureIgnoreCase))
                    {
                        seqGenMetric = new BleuMetric();
                    }
                    else
                    {
                        seqGenMetric = new RougeMetric();
                    }
                    Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
                    List<IMetric> task1Metrics = new List<IMetric>
                    {
                        seqGenMetric,
                        new LengthRatioMetric()
                    };

                    taskId2metrics.Add(1, task1Metrics);

                    List<IMetric> task0Metrics = new List<IMetric>()
                        {
                            new MultiLabelsFscoreMetric("", ss.ClsVocab.GetAllTokens(keepBuildInTokens: false))
                        };
                    taskId2metrics.Add(0, task0Metrics);


                    ss.Valid(validCorpus: validCorpus, taskId2metrics);
                }
                else if ( opts.Task == ModeEnums.Test )
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new Seq2SeqClassification(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    ss.Test<Seq2SeqClassificationCorpusBatch>(opts.InputTestFile, opts.OutputFile, opts.BatchSize, opts.MaxTestSrcSentLength);
                    
                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if ( opts.Task == ModeEnums.DumpVocab )
                {
                    ss = new Seq2SeqClassification(opts);
                    ss.DumpVocabToFiles(opts.SrcVocab, opts.TgtVocab, opts.ClsVocab);
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

        private static void ShowOptions(string[] args, Seq2SeqClassificationOptions opts)
        {
            string commandLine = string.Join(" ", args);
            string strOpts = JsonConvert.SerializeObject( opts, Formatting.Indented, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore, Converters = new[] { new StringEnumConverter() }, } );
            Logger.WriteLine($"Seq2SeqSharp v2.3.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
