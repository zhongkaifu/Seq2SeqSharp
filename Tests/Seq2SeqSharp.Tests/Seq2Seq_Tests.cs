// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.LearningRate;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Seq2SeqSharp.Enums;

namespace Seq2SeqSharp.Tests;

[TestClass]
[DeploymentItem("train.enu.snt")]
[DeploymentItem("train.chs.snt")]
public class Seq2Seq_Tests
{
    private string trainFolderPath = String.Empty;
    private string validFolderPath = String.Empty;

    [TestInitialize]
    public void TestInitialize()
    {
        // Prepare data set for training
        string rootPath = Directory.GetCurrentDirectory();
        trainFolderPath = rootPath; // Path.Combine(rootPath, "train");
        validFolderPath = rootPath; // Path.Combine(rootPath, "valid");

        if (Directory.Exists(trainFolderPath) == false)
        {
            Directory.CreateDirectory(trainFolderPath);
        }

        if (Directory.Exists(validFolderPath) == false)
        {
            Directory.CreateDirectory(validFolderPath);
        }

        //// Using same data set for both training and validation -- Just for testing.
        //File.Copy("train.enu.snt", Path.Combine(trainFolderPath, "train.enu.snt"), true);
        //File.Copy("train.chs.snt", Path.Combine(trainFolderPath, "train.chs.snt"), true);

        //File.Copy("train.enu.snt", Path.Combine(validFolderPath, "valid.enu.snt"), true);
        //File.Copy("train.chs.snt", Path.Combine(validFolderPath, "valid.chs.snt"), true);

    }

    //[TestMethod]
    //[DeploymentItem("mt_enu_chs.model")]
    //public void TestSeq2SeqInference()
    //{
    //    var opts = new Seq2SeqOptions();
    //    opts.ModelFilePath = "mt_enu_chs.model";
    //    opts.MaxSrcSentLength = 110;
    //    opts.MaxTgtSentLength = 110;
    //    opts.ProcessorType = ProcessorTypeEnums.CPU;
    //    opts.DeviceIds = "0";

    //    var seq2seq = new Seq2Seq(opts);
    //    DecodingOptions decodingOptions = opts.CreateDecodingOptions();

    //    List<List<List<string>>> groupBatchTokens = BuildInputGroupBatchTokens("▁Would ▁you ▁do ▁me ▁a ▁favor ▁by ▁moving ▁that ▁chair ▁over ▁to ▁the ▁corner ▁so ▁that ▁I ▁can ▁sweep ▁the ▁floor ?");
    //    var nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, null, decodingOptions);
    //    var out_tokens = nrs[0].Output[0][0];
    //    var output = string.Join(" ", out_tokens);
    //    Assert.IsTrue(output == "<s> ▁你会 帮我 一个 忙 , 把 椅子 挪 到 角落 里 , 这样 我 就可以 把 地板 吗 ? </s>");


    //    groupBatchTokens = BuildInputGroupBatchTokens("▁As ▁the ▁saying ▁goes : ▁\" good ▁always ▁triumph s ▁over ▁evil \". ▁But ▁there ' s ▁another ▁saying ▁that : ▁\" the ▁fight ▁against ▁evil ▁is ▁a ▁constant ▁struggle \".");
    //    nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, null, decodingOptions);
    //    out_tokens = nrs[0].Output[0][0];
    //    output = string.Join(" ", out_tokens);
    //    Assert.IsTrue(output == "<s> ▁正如 谚 语 所说 :“ 善 人 总是 战胜 邪恶 ”。 但 又 有 另一个 说 : ” 与 恶 作 斗争 是一 次 持续的 斗争 。 </s>");

    //}

    //[TestMethod]
    //[DeploymentItem("mt_enu_chs.model")]
    //public void TestSeq2SeqInferenceWithPrompt()
    //{
    //    var opts = new Seq2SeqOptions();
    //    opts.ModelFilePath = "mt_enu_chs.model";
    //    opts.MaxSrcSentLength = 110;
    //    opts.MaxTgtSentLength = 110;
    //    opts.ProcessorType = ProcessorTypeEnums.CPU;
    //    opts.DeviceIds = "0";

    //    var seq2seq = new Seq2Seq(opts);
    //    DecodingOptions decodingOptions = opts.CreateDecodingOptions();

    //    List<List<List<string>>> groupBatchTokens = BuildInputGroupBatchTokens("▁Would ▁you ▁do ▁me ▁a ▁favor ▁by ▁moving ▁that ▁chair ▁over ▁to ▁the ▁corner ▁so ▁that ▁I ▁can ▁sweep ▁the ▁floor ?");
    //    List<List<List<string>>> promptGroupBatchTokens = BuildInputGroupBatchTokens("<s> ▁ 你是否");

    //    var nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, promptGroupBatchTokens, decodingOptions);
    //    var out_tokens = nrs[0].Output[0][0];
    //    var output = string.Join(" ", out_tokens);
    //    Assert.IsTrue(output == "<s> ▁ 你是否 能 帮我 一个 忙 , 把 椅子 挪 到 角落 里 , 这样 我 就能 把 地板 。 </s>");


    //    groupBatchTokens = BuildInputGroupBatchTokens("▁luxury ▁for ▁the ▁skin ▁with ▁an ▁exceptionally ▁smooth ▁look");
    //    promptGroupBatchTokens = BuildInputGroupBatchTokens("肌 肤");
    //    nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, promptGroupBatchTokens, decodingOptions);
    //    out_tokens = nrs[0].Output[0][0];
    //    output = string.Join(" ", out_tokens);
    //    Assert.IsTrue(output == "<s> 肌 肤 的 奢 华 , 带 一 键 光 滑 的 眼 光 </s>");

    //}

    //[TestMethod]
    //[DeploymentItem("tinybook.model")]
    //public void TestGPTInferenceWithPrompt()
    //{
    //    var opts = new Seq2SeqOptions();
    //    opts.ModelFilePath = "tinybook.model";
    //    opts.MaxSrcSentLength = 32;
    //    opts.MaxTgtSentLength = 64;
    //    opts.ProcessorType = ProcessorTypeEnums.CPU;
    //    opts.DeviceIds = "0";
    //    opts.DecodingStrategy = DecodingStrategyEnums.Sampling;
    //    opts.DecodingTopP = 0.0f;

    //    //We use a fixed random seed.
    //    RandomGenerator.Init(8032023);

    //    var gpt = new GPT(opts);
    //    DecodingOptions decodingOptions = opts.CreateDecodingOptions();

    //    List<List<List<string>>> groupBatchTokens = BuildInputGroupBatchTokens("肌 肤 的 奢 华");

    //    var nrs = gpt.Test<SeqCorpusBatch>(groupBatchTokens, groupBatchTokens, decodingOptions);
    //    var out_tokens = nrs[0].Output[0][0];
    //    var output = string.Join(" ", out_tokens);
    //   Assert.IsTrue(output == "<s> 肌 肤 的 奢 华 , 让 刘 备 很 是 惊讶 。 ▁ 刘 备 不 禁 问道 :“ 杨 广 的 弟弟 杨 家 , 刘 备 可 听说过 ?” ▁ 刘 备 摇 头 道 :“ 杨 家 杨 过 在 刘 邦 时期 , 曾在 刘 赵 两国 中 原 。 ” ▁ 杨 广 不 禁 问道 :“");
    //}


    private static List<List<List<string>>> BuildInputGroupBatchTokens(string input)
    {
        var tokens = input.Split(' ').ToList();
        var batchTokens = new List<List<string>> { tokens };
        var groupBatchTokens = new List<List<List<string>>> { batchTokens };
        return groupBatchTokens;
    }

    static double lastEpochAvgCost = 1000000.0;


    [TestMethod]
    public void TestSeq2SeqBuildSharedVocabs()
    {
        // Build configs for training
        Seq2SeqOptions opts = CreateOptions(trainFolderPath, validFolderPath);

        // Load training corpus
        var trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
            maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence);

        // Build vocabularies for training
        (var srcVocab, var tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize, sharedVocab: true);

        Assert.IsTrue(srcVocab.Count == tgtVocab.Count);

        var srcTokens = srcVocab.GetAllTokens();
        var tgtTokens = tgtVocab.GetAllTokens();

        for (int i = 0; i < srcTokens.Count; i++)
        {
            Assert.IsTrue(srcTokens[i] == tgtTokens[i]);
        }
    }


    [TestMethod]
    public void TestSeq2SeqCorpusPrefixSuffix()
    {
        // Build configs for training
        Seq2SeqOptions opts = CreateOptions(trainFolderPath, validFolderPath);

        // Load training corpus
        var trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
            maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence);

        foreach (var batch in trainCorpus)
        {
            var srcBatch = batch.GetSrcTokens();
            foreach (var srcTokens in srcBatch)
            {
                Assert.IsTrue(srcTokens[0] == BuildInTokens.BOS);
                Assert.IsTrue(srcTokens[srcTokens.Count - 1] == BuildInTokens.EOS);
            }

            var tgtBatch = batch.GetTgtTokens();
            foreach (var tgtTokens in tgtBatch)
            {
                Assert.IsTrue(tgtTokens[0] == BuildInTokens.BOS);
                Assert.IsTrue(tgtTokens[tgtTokens.Count - 1] == BuildInTokens.EOS);
            }
        }

    }


    [TestMethod]
    public void TestSeq2SeqTraining()
    {
        // Build configs for training
        Seq2SeqOptions opts = CreateOptions(trainFolderPath, validFolderPath);

        DecodingOptions decodingOptions = opts.CreateDecodingOptions();

        // Load training corpus
        var trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
            maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence);

        // Load valid corpus
        var validCorpusList = new List<Seq2SeqCorpus>();
        if (!opts.ValidCorpusPaths.IsNullOrEmpty())
        {
            string[] validCorpusPathList = opts.ValidCorpusPaths.Split(';');
            foreach (var validCorpusPath in validCorpusPathList)
            {
                validCorpusList.Add(new Seq2SeqCorpus(validCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValMaxTokenSizePerBatch, opts.MaxValidSrcSentLength, opts.MaxValidTgtSentLength, shuffleEnums: opts.ShuffleType, tooLongSequence: opts.TooLongSequence));
            }
        }

        // Create learning rate
        ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount, opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

        // Create optimizer
        IOptimizer optimizer = Misc.CreateOptimizer(opts);

        // Build vocabularies for training
        (var srcVocab, var tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize, opts.SharedEmbeddings);

        // Create metrics
        List<IMetric> metrics = new List<IMetric> { new BleuMetric() };

        //New training
        var ss = new Seq2Seq(opts, srcVocab, tgtVocab);

        // Add event handler for monitoring       
        ss.StatusUpdateWatcher += Ss_StatusUpdateWatcher;
        ss.EpochEndWatcher += Ss_EpochEndWatcher;

        // Kick off training
        ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpusList: validCorpusList.ToArray(), learningRate: learningRate, optimizer: optimizer, metrics: metrics.ToArray(), decodingOptions: decodingOptions);

        ss.SaveModel(suffix: ".test");

        // Check if model file exists
        Assert.IsTrue(File.Exists(opts.ModelFilePath + ".test"));
    }

    public static void Ss_StatusUpdateWatcher(object? sender, EventArgs e)
    {
        CostEventArg? ep = e as CostEventArg;
        if (ep != null)
        {
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

            Assert.IsFalse(double.IsNaN(ep.AvgCostInTotal));
        }
        else
        {
            throw new ArgumentNullException("The input event argument e is not a CostEventArg.");
        }
    }

    public static void Ss_EpochEndWatcher(object? sender, EventArgs e)
    {
        CostEventArg? ep = e as CostEventArg;

        if (ep != null)
        {
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

            Assert.IsFalse(double.IsNaN(ep.AvgCostInTotal));

            Assert.IsTrue(ep.AvgCostInTotal < lastEpochAvgCost);
            lastEpochAvgCost = ep.AvgCostInTotal;
        }
        else
        {
            throw new ArgumentNullException("The input event argument e is not a CostEventArg.");
        }
    }

    private static Seq2SeqOptions CreateOptions(string trainFolderPath, string validFolderPath)
    {
        Seq2SeqOptions opts = new Seq2SeqOptions();
        opts.Task = ModeEnums.Train;

        opts.TrainCorpusPath = trainFolderPath;
        opts.ValidCorpusPaths = validFolderPath;
        opts.SrcLang = "ENU";
        opts.TgtLang = "CHS";

        opts.EncoderLayerDepth = 2;
        opts.DecoderLayerDepth = 2;
        opts.SrcEmbeddingDim = 64;
        opts.TgtEmbeddingDim = 64;
        opts.HiddenSize = 64;
        opts.MultiHeadNum = 8;

        opts.StartLearningRate = 0.0006f;
        opts.WarmUpSteps = 8000;
        opts.WeightsUpdateCount = 8000;

        opts.MaxTokenSizePerBatch = 128;
        opts.ValMaxTokenSizePerBatch = 128;
        opts.MaxSrcSentLength = 110;
        opts.MaxTgtSentLength = 110;
        opts.MaxValidSrcSentLength = 110;
        opts.MaxValidTgtSentLength = 110;
        opts.ShuffleType = Utils.ShuffleEnums.NoPadding;
        opts.TooLongSequence = TooLongSequence.Truncation;
        opts.ProcessorType = ProcessorTypeEnums.CPU;
        opts.MaxEpochNum = 3;

        opts.ModelFilePath = "seq2seq_test.model";

        return opts;
    }
}
