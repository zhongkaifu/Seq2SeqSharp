using Microsoft.VisualStudio.TestTools.UnitTesting;
using Seq2SeqSharp.Tools;

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
}