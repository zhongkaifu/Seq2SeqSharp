using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SeqLabelConsole
{
    class Options
    {
        [Arg("Task name. It could be Train, Valid, Test, VisualizeNetwork or Help", "TaskName")]
        public string TaskName = "Help";

        [Arg("The vector size of encoded source word.", "WordVectorSize")]
        public int WordVectorSize = 128;

        [Arg("The hidden layer size of encoder and decoder.", "HiddenSize")]
        public int HiddenSize = 128;

        [Arg("Start Learning rate.", "StartLearningRate")]
        public float StartLearningRate = 0.0003f;

        [Arg("The number of updates for weights", "WeightsUpdateCount")]
        public int WeightsUpdateCount = 0;

        [Arg("The network depth in encoder.", "EncoderLayerDepth")]
        public int EncoderLayerDepth = 1;

        [Arg("The trained model file path.", "ModelFilePath")]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The vocabulary file path for source side.", "SrcVocab")]
        public string SrcVocab = null;

        [Arg("The vocabulary file path for target side.", "TgtVocab")]
        public string TgtVocab = null;

        [Arg("The external embedding model file path for source side.", "SrcEmbedding")]
        public string SrcEmbeddingModelFilePath = null;

        [Arg("Source language name.", "SrcLang")]
        public string SrcLang;

        [Arg("Target language name.", "TgtLang")]
        public string TgtLang;

        [Arg("Training corpus folder path", "TrainCorpusPath")]
        public string TrainCorpusPath;

        [Arg("Valid corpus folder path", "ValidCorpusPath")]
        public string ValidCorpusPath;

        [Arg("The input file for test.", "InputTestFile")]
        public string InputTestFile;

        [Arg("The test result file.", "OutputTestFile")]
        public string OutputTestFile;

        [Arg("The shuffle block size", "ShuffleBlockSize")]
        public int ShuffleBlockSize = -1;

        [Arg("Clip gradients", "GradClip")]
        public float GradClip = 3.0f;

        [Arg("The batch size", "BatchSize")]
        public int BatchSize = 1;

        [Arg("Dropout ratio", "Dropout")]
        public float DropoutRatio = 0.1f;

        [Arg("Processor type: GPU, CPU", "ProcessorType")]
        public string ProcessorType = "GPU";

        [Arg("Encoder type: BiLSTM, Transformer", "EncoderType")]
        public string EncoderType = "BiLSTM";

        [Arg("The number of multi-heads in transformer model", "MultiHeadNum")]
        public int MultiHeadNum = 8;

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", "DeviceIds")]
        public string DeviceIds = "0";

        [Arg("Beam search size. Default is 1", "BeamSearch")]
        public int BeamSearch = 1;

        [Arg("Maxmium epoch number during training. Default is 100", "MaxEpochNum")]
        public int MaxEpochNum = 100;

        [Arg("Maxmium sentence length", "MaxSentLength")]
        public int MaxSentLength = 32;

        [Arg("The number of steps for warming up", "WarmUpSteps")]
        public int WarmUpSteps = 8000;

        [Arg("The file path of dumped neural network (.png file)", "VisNNFile")]
        public string VisualizeNNFilePath;

        [Arg("The file path of config file for parameters", "ConfigFilePath")]
        public string ConfigFilePath = String.Empty;
    }
}
