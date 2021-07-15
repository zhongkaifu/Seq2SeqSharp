using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class Options
    {
        [Arg("The batch size", "BatchSize")]
        public int BatchSize = 1;

        [Arg("Beam search size. Default is 1", "BeamSearchSize")]
        public int BeamSearchSize = 1;

        [Arg("The beta1 for optimizer", "Beta1")]
        public float Beta1 = 0.9f;

        [Arg("The beta2 for optimizer", "Beta2")]
        public float Beta2 = 0.98f;

        [Arg("The options for CUDA NVRTC compiler. Options are split by space. For example: \"--use_fast_math --gpu-architecture=compute_60\"", "CompilerOptions")]
        public string CompilerOptions = "--use_fast_math";

        [Arg("The file path of config file for parameters", "ConfigFilePath")]
        public string ConfigFilePath = string.Empty;

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", "DeviceIds")]
        public string DeviceIds = "0";

        [Arg("Dropout ratio", "Dropout")]
        public float DropoutRatio = 0.0f;

        [Arg("Enable segment embeddings", "EnableSegmentEmbeddings")]
        public bool EnableSegmentEmbeddings = false;

        [Arg("The network depth in encoder.", "EncoderLayerDepth")]
        public int EncoderLayerDepth = 1;

        [Arg("Starting Learning rate factor for encoders", "EncoderStartLearningRateFactor")]
        public float EncoderStartLearningRateFactor = 1.0f;

        [Arg("Encoder type: LSTM, BiLSTM, Transformer", "EncoderType")]
        public string EncoderType = "Transformer";

        [Arg("Clip gradients", "GradClip")]
        public float GradClip = 3.0f;

        [Arg("The hidden layer size of encoder and decoder.", "HiddenSize")]
        public int HiddenSize = 128;

        [Arg("The input file for test.", "InputTestFile")]
        public string InputTestFile = null;

        [Arg("Starting Learning rate", "StartLearningRate")]
        public float StartLearningRate = 0.0006f;

        [Arg("Maxmium epoch number during training. Default is 100", "MaxEpochNum")]
        public int MaxEpochNum = 100;

        [Arg("The ratio of memory usage", "MemoryUsageRatio")]
        public float MemoryUsageRatio = 0.95f;

        [Arg("The trained model file path.", "ModelFilePath")]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The number of multi-heads in transformer model", "MultiHeadNum")]
        public int MultiHeadNum = 8;

        [Arg("The email to notify evaluation result", "NotifyEmail")]
        public string NotifyEmail = "";

        [Arg("The weights optimizer during training. It supports Adam and RMSProp. Adam is default", "Optimizer")]
        public string Optimizer = "Adam";

        [Arg("The test result file.", "OutputFile")]
        public string OutputFile = null;

        [Arg("Processor type: GPU, CPU", "ProcessorType")]
        public string ProcessorType = "GPU";

        [Arg("The vocabulary file path for source side.", "SrcVocab")]
        public string SrcVocab = null;

        [Arg("The shuffle block size", "ShuffleBlockSize")]
        public int ShuffleBlockSize = -1;

        [Arg("Type of shuffling. It could be NoPaddingInSrc, NoPaddingInTgt and Random", "ShuffleType")]
        public string ShuffleType = "Random";

        [Arg("Task to execute. It could be Train, Valid, Test, DumpVocab or Help", "Task")]
        public string Task = "Help";

        [Arg("The vocabulary file path for target side.", "TgtVocab")]
        public string TgtVocab = null;

        [Arg("Training corpus folder path", "TrainCorpusPath")]
        public string TrainCorpusPath = null;

        [Arg("The batch size during validation", nameof(ValBatchSize))]
        public int ValBatchSize = 1;

        [Arg("Valid corpus folder path", "ValidCorpusPath")]
        public string ValidCorpusPath = null;

        [Arg("The number of steps for warming up", "WarmUpSteps")]
        public int WarmUpSteps = 8000;

        [Arg("The number of updates for weights", "WeightsUpdateCount")]
        public int WeightsUpdateCount = 0;

        [Arg("The interval hours to run model validation", "ValidIntervalHours")]
        public float ValidIntervalHours = 1.0f;

        [Arg("The size of vocabulary", "VocabSize")]
        public int VocabSize = 45000;
    }
}
