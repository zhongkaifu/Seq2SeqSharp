using AdvUtils;
using Newtonsoft.Json;
using Seq2SeqSharp;
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

namespace Seq2SeqConsole
{
    internal class Program
    {
        private static Seq2SeqOptions opts = new Seq2SeqOptions();
        private static void ss_EvaluationWatcher(object sender, EventArgs e)
        {
            EvaluationEventArg ep = e as EvaluationEventArg;
            Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);

            if (String.IsNullOrEmpty(opts.NotifyEmail) == false)
            {
                Email.Send(ep.Title, ep.Message, opts.NotifyEmail, new string[] { opts.NotifyEmail });
            }
        }

        private static void ss_StatusUpdateWatcher(object sender, EventArgs e)
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

            Logger.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate.ToString("F6")}, AvgCost = {ep.AvgCostInTotal.ToString("F4")}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin.ToString("F")}, WordPerSec = {wordPerSec.ToString("F")}");
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
                    opts = JsonConvert.DeserializeObject<Seq2SeqOptions>(File.ReadAllText(opts.ConfigFilePath));
                }

                Logger.LogFile = $"{nameof(Seq2SeqConsole)}_{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log";
                ShowOptions(args, opts);

                Seq2Seq ss = null;
                ModeEnums mode = (ModeEnums)Enum.Parse(typeof(ModeEnums), opts.Task);
                ShuffleEnums shuffleType = (ShuffleEnums)Enum.Parse(typeof(ShuffleEnums), opts.ShuffleType);

                if (mode == ModeEnums.Train)
                {
                    // Load train corpus
                    Seq2SeqCorpus trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize, shuffleBlockSize: opts.ShuffleBlockSize,
                        maxSrcSentLength: opts.MaxTrainSrcSentLength, maxTgtSentLength: opts.MaxTrainTgtSentLength, shuffleEnums: shuffleType);
                    // Load valid corpus
                    Seq2SeqCorpus validCorpus = string.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new Seq2SeqCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: shuffleType);

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
                    List<IMetric> metrics = new List<IMetric>
                    {
                        seqGenMetric,
                        new LengthRatioMetric()
                    };

                    if (!String.IsNullOrEmpty(opts.ModelFilePath) && File.Exists(opts.ModelFilePath))
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
                        if (!string.IsNullOrEmpty(opts.SrcVocab) && !string.IsNullOrEmpty(opts.TgtVocab))
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

                            (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(opts.VocabSize, opts.SharedEmbeddings);
                        }

                        //New training
                        ss = new Seq2Seq(opts, srcVocab, tgtVocab);
                    }

                    // Add event handler for monitoring
                    ss.StatusUpdateWatcher += ss_StatusUpdateWatcher;
                    ss.EvaluationWatcher += ss_EvaluationWatcher;

                    // Kick off training
                    ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, metrics: metrics);
                }
                else if (mode == ModeEnums.Valid)
                {
                    Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPath}'");

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
                    List<IMetric> metrics = new List<IMetric>
                {
                        seqGenMetric,
                    new LengthRatioMetric()
                    };

                    // Load valid corpus
                    Seq2SeqCorpus validCorpus = new Seq2SeqCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxTestSrcSentLength, opts.MaxTestTgtSentLength, shuffleEnums: shuffleType);

                    ss = new Seq2Seq(opts);
                    ss.EvaluationWatcher += ss_EvaluationWatcher;
                    ss.Valid(validCorpus: validCorpus, metrics: metrics);
                }
                else if (mode == ModeEnums.Test)
                {
                    Logger.WriteLine($"Test model: '{opts.ModelFilePath}'");
                    Logger.WriteLine($"Test set: '{opts.InputTestFile}'");
                    Logger.WriteLine($"Max source sentence length: '{opts.MaxTestSrcSentLength}'");
                    Logger.WriteLine($"Max target sentence length: '{opts.MaxTestTgtSentLength}'");
                    Logger.WriteLine($"Beam search size: '{opts.BeamSearchSize}'");
                    Logger.WriteLine($"Batch size: '{opts.BatchSize}'");
                    Logger.WriteLine($"Shuffle type: '{opts.ShuffleType}'");
                    Logger.WriteLine($"Device ids: '{opts.DeviceIds}'");


                    if (File.Exists(opts.OutputFile))
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
                        File.Delete(opts.OutputFile);
                    }

                    //Test trained model
                    ss = new Seq2Seq(opts);
                    List<List<List<string>>> inputBatchs = new List<List<List<string>>>(); // shape: (feature_group_size, batch_size, sequence_length)
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    foreach (string line in File.ReadLines(opts.InputTestFile))
                    {
                        Misc.AppendNewBatch(inputBatchs, line, opts.MaxTestSrcSentLength);

                        if (inputBatchs[0].Count >= opts.BatchSize * ss.DeviceIds.Length)
                        {
                            (var outputLines, var alignments) = RunBatchTest(opts, ss, inputBatchs);
                            File.AppendAllLines(opts.OutputFile, outputLines);
                            if (opts.OutputAlignment)
                            {
                                File.AppendAllLines(opts.OutputFile + ".alignment", alignments);
                            }

                            inputBatchs.Clear();
                        }
                    }

                    if (inputBatchs.Count > 0 && inputBatchs[0].Count > 0)
                    {
                        (var outputLines, var alignments) = RunBatchTest(opts, ss, inputBatchs);
                        File.AppendAllLines(opts.OutputFile, outputLines);
                        if (opts.OutputAlignment)
                        {
                            File.AppendAllLines(opts.OutputFile + ".alignment", alignments);
                        }
                    }
                    stopwatch.Stop();

                    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");
                }
                else if (mode == ModeEnums.DumpVocab)
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


        private static (List<string>, List<string>) RunBatchTest(Seq2SeqOptions opts, Seq2Seq ss, List<List<List<string>>> inputBatchs)
        {
            List<string> outputLines = new List<string>();
            List<string> alignments = new List<string>();

            NetworkResult nr = ss.Test(inputBatchs, opts.BeamSearchSize); // shape [beam size, batch size, tgt token size]

            for (int batchIdx = 0; batchIdx < inputBatchs[0].Count; batchIdx++)
            {
                for (int beamIdx = 0; beamIdx < nr.Output.Count; beamIdx++)
                {
                    outputLines.Add(String.Join(" ", nr.Output[beamIdx][batchIdx]));

                    if (opts.OutputAlignment)
                    {
                        StringBuilder sb = new StringBuilder();
                        for (int tgtTknIdx = 0; tgtTknIdx < nr.Output[beamIdx][batchIdx].Count; tgtTknIdx++)
                        {
                            int srcIdx = nr.Alignment[beamIdx][batchIdx][tgtTknIdx].SrcPos;
                            float score = nr.Alignment[beamIdx][batchIdx][tgtTknIdx].Score;
                            sb.Append($"{tgtTknIdx}_{srcIdx}_{score}");
                            sb.Append(" ");
                        }

                        alignments.Add(sb.ToString().Trim());
                    }
                }
            }

            return (outputLines, alignments);
        }

        private static void ShowOptions(string[] args, Seq2SeqOptions opts)
        {
            string commandLine = string.Join(" ", args);
            string strOpts = JsonConvert.SerializeObject(opts);

            Logger.WriteLine($"Seq2SeqSharp v2.3.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");
            Logger.WriteLine($"Command Line = '{commandLine}'");
            Logger.WriteLine($"Configs: {strOpts}");
        }
    }
}
