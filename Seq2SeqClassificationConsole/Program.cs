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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace Seq2SeqClassificationConsole
{
    internal class Program
    {
        private static Seq2SeqClassificationOptions opts = new Seq2SeqClassificationOptions();
        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (String.IsNullOrEmpty(opts.NotifyEmail) == false)
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void Ss_StatusUpdateWatcher(object sender, EventArgs e)
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
            try
            {                               
                //Parse command line
                ArgParser argParser = new ArgParser(args, opts);

                if (string.IsNullOrEmpty(opts.ConfigFilePath) == false)
                {
                    Console.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    opts = JsonConvert.DeserializeObject<Seq2SeqClassificationOptions>(File.ReadAllText(opts.ConfigFilePath));
                }

                Logger.LogFile = $"{nameof(Seq2SeqClassificationConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";
                ShowOptions(args, opts);

                Seq2SeqClassification ss = null;
                ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.Task);
                ShuffleEnums shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), opts.ShuffleType);

                if (mode == ModeEnums.Train)
                {
                    // Load train corpus
                    Seq2SeqClassificationCorpus trainCorpus = new Seq2SeqClassificationCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize, shuffleBlockSize: opts.ShuffleBlockSize,
                        maxSrcSentLength: opts.MaxTrainSrcSentLength, maxTgtSentLength: opts.MaxTrainTgtSentLength, shuffleEnums: shuffleType);
                    // Load valid corpus
                    Seq2SeqClassificationCorpus validCorpus = string.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new Seq2SeqClassificationCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: shuffleType);

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


                    List<IMetric> task0Metrics = new List<IMetric>();
                    if (!String.IsNullOrEmpty(opts.ModelFilePath) && File.Exists(opts.ModelFilePath))
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
                        if (!string.IsNullOrEmpty(opts.SrcVocab) && !string.IsNullOrEmpty(opts.TgtVocab) && !string.IsNullOrEmpty(opts.ClsVocab))
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

                            if (!String.IsNullOrEmpty(opts.SrcVocab))
                            {
                                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Source vocabulary '{opts.SrcVocab}' is not empty, but we will build it from training corpus.");
                            }

                            if (!String.IsNullOrEmpty(opts.TgtVocab))
                            {
                                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Target vocabulary '{opts.TgtVocab}' is not empty, but we will build it from training corpus.");
                            }

                            if (!String.IsNullOrEmpty(opts.ClsVocab))
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
                    ss.StatusUpdateWatcher += Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, taskId2metrics: taskId2metrics);
                }
                else if (mode == ModeEnums.Valid)
                {
                    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'");

                    // Load valid corpus
                    Seq2SeqClassificationCorpus validCorpus = new Seq2SeqClassificationCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: shuffleType);

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
                else if (mode == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new Seq2SeqClassification(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    ss.Test();
                    
                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (mode == ModeEnums.DumpVocab)
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
            string strOpts = JsonConvert.SerializeObject(opts);
            Logger.WriteLine($"Seq2SeqSharp v2.3.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
