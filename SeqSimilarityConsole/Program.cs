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

namespace SeqSimilarityConsole
{
    class Program
    {
        private static SeqSimilarityOptions opts = new SeqSimilarityOptions();
        private static void Ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (String.IsNullOrEmpty(opts.NotifyEmail) == false)
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void ShowOptions(string[] args, SeqSimilarityOptions opts)
        {
            string commandLine = string.Join(" ", args);
            Logger.WriteLine($"SeqSimilarityConsole v2.3.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");

            string strOpts = JsonConvert.SerializeObject(opts);
            Logger.WriteLine($"Configs: {strOpts}");
        }

        static void Main(string[] args)
        {
            try
            {
                //Parse command line
                //   Seq2SeqOptions opts = new Seq2SeqOptions();
                ArgParser argParser = new ArgParser(args, opts);

                if (string.IsNullOrEmpty(opts.ConfigFilePath) == false)
                {
                    Logger.WriteLine($"Loading config file from '{opts.ConfigFilePath}'");
                    opts = JsonConvert.DeserializeObject<SeqSimilarityOptions>(File.ReadAllText(opts.ConfigFilePath));
                }

                Logger.LogFile = $"{nameof(SeqSimilarityConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";
                ShowOptions(args, opts);

                SeqSimilarity ss = null;
                ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.Task);
                ShuffleEnums shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), opts.ShuffleType);

                if (mode == ModeEnums.Train)
                {
                    // Load train corpus
                    SeqClassificationMultiTasksCorpus trainCorpus = new SeqClassificationMultiTasksCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize, shuffleBlockSize: opts.ShuffleBlockSize,
                        maxSentLength: opts.MaxTrainSentLength, shuffleEnums: shuffleType);
                    // Load valid corpus
                    SeqClassificationMultiTasksCorpus validCorpus = string.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new SeqClassificationMultiTasksCorpus(opts.ValidCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSentLength, shuffleEnums: shuffleType);

                    // Create learning rate
                    ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                    // Create metrics
                    Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();

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


                    if (!String.IsNullOrEmpty(opts.ModelFilePath) && File.Exists(opts.ModelFilePath))
                    {
                        //Incremental training
                        Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                        ss = new SeqSimilarity(opts);

                            taskId2metrics.Add(0, new List<IMetric>());
                            taskId2metrics[0].Add(new MultiLabelsFscoreMetric("", ss.ClsVocab.GetAllTokens(keepBuildInTokens: false)));
                        
                    }
                    else
                    {
                        // Load or build vocabulary
                        Vocab srcVocab = null;
                        List<Vocab> tgtVocabs = null;
                        if (!string.IsNullOrEmpty(opts.SrcVocab) && !string.IsNullOrEmpty(opts.TgtVocab))
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
                        ss = new SeqSimilarity(opts, srcVocab, tgtVocabs[0]);
                    }

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += Ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, taskId2metrics: taskId2metrics);
                }
                //else if (mode == ModeEnums.Valid)
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
                else if (mode == ModeEnums.Test)
                {
                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new SeqSimilarity(opts);
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    ss.Test();

                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                //else if (mode == ModeEnums.DumpVocab)
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
