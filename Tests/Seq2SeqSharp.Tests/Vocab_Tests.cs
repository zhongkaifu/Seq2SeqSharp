using Microsoft.VisualStudio.TestTools.UnitTesting;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tests
{
    [TestClass]
    [DeploymentItem("enuSpm.vocab")]
    public class Vocab_Tests
    {
        [TestMethod]
        public void TestVocabBuildInTokens()
        {
            Vocab vocab = new Vocab("enuSpm.vocab");

            Assert.IsTrue((int)SENTTAGS.START == vocab.GetWordIndex(BuildInTokens.BOS));
            Assert.IsTrue((int)SENTTAGS.END == vocab.GetWordIndex(BuildInTokens.EOS));
            Assert.IsTrue((int)SENTTAGS.UNK == vocab.GetWordIndex(BuildInTokens.UNK));

            Assert.IsTrue(vocab.GetString((int)SENTTAGS.START) == BuildInTokens.BOS);
            Assert.IsTrue(vocab.GetString((int)SENTTAGS.END) == BuildInTokens.EOS);
            Assert.IsTrue(vocab.GetString((int)SENTTAGS.UNK) == BuildInTokens.UNK);
        }
    }
}
