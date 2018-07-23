using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Seq2SeqSharp;
using AdvUtils;

namespace Seq2SeqConsole
{
    class Options
    {
        [Arg("Task name. It could be train and test", "TaskName")]
        public string TaskName;

        [Arg("The vector size of encoded source word.", "WordVectorSize")]
        public int WordVectorSize = 128;

        [Arg("The hidden layer size of encoder and decoder.", "HiddenSize")]
        public int HiddenSize = 128;

        [Arg("Learning rate.", "LearningRate")]
        public float LearningRate = 0.001f;

        [Arg("The network depth in decoder.", "Depth")]
        public int Depth = 1;

        [Arg("The trained model file path.", "ModelFilePath")]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The vocabulary file path for source side.", "SrcVocab")]
        public string SrcVocab = null;

        [Arg("The vocabulary file path for target side.", "TgtVocab")]
        public string TgtVocab = null;

        [Arg("The external embedding model file path for source side.", "SrcEmbedding")]
        public string SrcEmbeddingModelFilePath = null;

        [Arg("The external embedding model file path for target side.", "TgtEmbedding")]
        public string TgtEmbeddingModelFilePath = null;

        [Arg("Source language name.", "SrcLang")]
        public string SrcLang;

        [Arg("Target language name.", "TgtLang")]
        public string TgtLang;

        [Arg("Training corpus folder path", "TrainCorpusPath")]
        public string TrainCorpusPath;

        [Arg("It indicates if sparse feature used for training.", "SparseFeature")]
        public bool SparseFeature = false;

        [Arg("The input file for test.", "InputTestFile")]
        public string InputTestFile;

        [Arg("The test result file.", "OutputTestFile")]
        public string OutputTestFile;

        [Arg("The shuffle block size", "ShuffleBlockSize")]
        public int ShuffleBlockSize = -1;


    }
}
