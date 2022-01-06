using Microsoft.VisualStudio.TestTools.UnitTesting;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp.Tests;

[TestClass]
public class ComputeGraph_Tests
{
    [TestMethod]
    public void TestAddTensorTensor()
    {
        TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

        var graph = new ComputeGraphTensor(new WeightTensorFactory(), 0, true);

        var tensorA = new WeightTensor(new long[2] { 2, 2 }, 1, 0, name: "tensorA", isTrainable: true);
        var tensorB = new WeightTensor(new long[2] { 2, 2 }, 2, 0, name: "tensorA", isTrainable: true);

        var tensorSum = graph.Add(tensorA, tensorB);

        float v = tensorSum.GetWeightAt(new long[] { 1, 1 });

        Assert.IsTrue(v == 3.0f);

    }

    [TestMethod]
    [DeploymentItem("seq2seq_mt_enu_chs_tiny_test.model")]
    public void TestSeq2SeqInference()
    {
        var opts = new Seq2SeqOptions();
        opts.ModelFilePath = "seq2seq_mt_enu_chs_tiny_test.model";
        opts.MaxTestSrcSentLength = 110;
        opts.MaxTestTgtSentLength = 110;
        opts.ProcessorType = ProcessorTypeEnums.CPU;
        opts.DeviceIds = "0";

        var seq2seq = new Seq2Seq(opts);
        DecodingOptions decodingOptions = opts.CreateDecodingOptions();

        List<List<List<string>>> groupBatchTokens = BuildInputGroupBatchTokens("▁yes , ▁solutions ▁do ▁exist .");
        var nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, null, decodingOptions);
        var out_tokens = nrs[0].Output[0][0];
        var output = string.Join(" ", out_tokens);
        Assert.IsTrue(output == "<s> ▁是的 , 解决方案 存在 。 </s>");


        groupBatchTokens = BuildInputGroupBatchTokens("▁a ▁question ▁of ▁climate .");
        nrs = seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens, null, decodingOptions);
        out_tokens = nrs[0].Output[0][0];
        output = string.Join(" ", out_tokens);
        Assert.IsTrue(output == "<s> ▁ 气候 问题 。 </s>");

    }

    private static List<List<List<string>>> BuildInputGroupBatchTokens(string input)
    {
        var tokens = input.Split(' ').ToList();
        var batchTokens = new List<List<string>> { tokens };
        var groupBatchTokens = new List<List<List<string>>> { batchTokens };
        return groupBatchTokens;
    }
}