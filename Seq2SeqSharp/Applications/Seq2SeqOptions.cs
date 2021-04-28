using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AdvUtils;

namespace Seq2SeqSharp
{
    public class Seq2SeqOptions
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

        [Arg("The network depth in decoder.", "DecoderLayerDepth")]
        public int DecoderLayerDepth = 1;

        [Arg("Decoder type: AttentionLSTM, Transformer", "DecoderType")]
        public string DecoderType = "Transformer";

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", "DeviceIds")]
        public string DeviceIds = "0";

        [Arg("Dropout ratio", "Dropout")]
        public float DropoutRatio = 0.1f;

        [Arg("Apply coverage model in decoder", "EnableCoverageModel")]
        public bool EnableCoverageModel = true;

        [Arg("Enable segment embeddings", "EnableSegmentEmbeddings")]
        public bool EnableSegmentEmbeddings = false;

        [Arg("The network depth in encoder.", "EncoderLayerDepth")]
        public int EncoderLayerDepth = 1;

        [Arg("Encoder type: LSTM, BiLSTM, Transformer", "EncoderType")]
        public string EncoderType = "Transformer";

        [Arg("The hidden layer size of encoder and decoder.", "HiddenSize")]
        public int HiddenSize = 128;

        [Arg("The input file for test.", "InputTestFile")]
        public string InputTestFile = null;

        [Arg("It indicates if the decoder is trainable", "IsDecoderTrainable")]
        public bool IsDecoderTrainable = true;

        [Arg("It indicates if the encoder is trainable", "IsEncoderTrainable")]
        public bool IsEncoderTrainable = true;

        [Arg("It indicates if the src embedding is trainable", "IsSrcEmbeddingTrainable")]
        public bool IsSrcEmbeddingTrainable = true;
        [Arg("It indicates if the tgt embedding is trainable", "IsTgtEmbeddingTrainable")]
        public bool IsTgtEmbeddingTrainable = true;

        [Arg("Clip gradients", "GradClip")]
        public float GradClip = 3.0f;

        [Arg("Maxmium epoch number during training. Default is 100", "MaxEpochNum")]
        public int MaxEpochNum = 100;

        [Arg("Maxmium src sentence length", "MaxSrcSentLength")]
        public int MaxSrcSentLength = 32;

        [Arg("Maxmium tgt sentence length", "MaxTgtSentLength")]
        public int MaxTgtSentLength = 32;

        [Arg("The ratio of memory usage", "MemoryUsageRatio")]
        public float MemoryUsageRatio = 0.95f;

        [Arg("The trained model file path.", "ModelFilePath")]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The number of multi-heads in transformer model", "MultiHeadNum")]
        public int MultiHeadNum = 8;

        [Arg("The weights optimizer during training. It supports Adam and RMSProp. Adam is default", "Optimizer")]
        public string Optimizer = "Adam";

        [Arg("It indicates if output alignment between target tokens and source tokens", "OutputAlignment")]
        public bool OutputAlignment = false;

        [Arg("The test result file.", "OutputFile")]
        public string OutputFile = null;

        [Arg("Processor type: GPU, CPU", "ProcessorType")]
        public string ProcessorType = "GPU";

        [Arg("Sharing embeddings between source side and target side", "SharedEmbeddings")]
        public bool SharedEmbeddings = false;

        [Arg("The shuffle block size", "ShuffleBlockSize")]
        public int ShuffleBlockSize = -1;

        [Arg("Type of shuffling. It could be NoPaddingInSrc, NoPaddingInTgt and Random", "ShuffleType")]
        public string ShuffleType = "Random";

        [Arg("The embedding dim in source side", "SrcEmbeddingDim")]
        public int SrcEmbeddingDim = 128;

        [Arg("The external embedding model file path for source side.", "SrcEmbedding")]
        public string SrcEmbeddingModelFilePath = null;

        [Arg("Source language name.", "SrcLang")]
        public string SrcLang;

        [Arg("The vocabulary file path for source side.", "SrcVocab")]
        public string SrcVocab = null;

        [Arg("Start Learning rate.", "StartLearningRate")]
        public float StartLearningRate = 0.0003f;

        [Arg("Task to execute. It could be Train, Valid, Test, DumpVocab or Help", "Task")]
        public string Task = "Help";

        [Arg("The embedding dim in target side", "TgtEmbeddingDim")]
        public int TgtEmbeddingDim = 128;

        [Arg("The external embedding model file path for target side.", "TgtEmbedding")]
        public string TgtEmbeddingModelFilePath = null;

        [Arg("Target language name.", "TgtLang")]
        public string TgtLang;

        [Arg("The vocabulary file path for target side.", "TgtVocab")]
        public string TgtVocab = null;

        [Arg("Training corpus folder path", "TrainCorpusPath")]
        public string TrainCorpusPath = null;

        [Arg("The batch size during validation", nameof(ValBatchSize))]
        public int ValBatchSize = 1;

        [Arg("Valid corpus folder path", "ValidCorpusPath")]
        public string ValidCorpusPath = null;

        [Arg("The size of vocabulary", "VocabSize")]
        public int VocabSize = 45000;

        [Arg("The number of steps for warming up", "WarmUpSteps")]
        public int WarmUpSteps = 8000;

        [Arg("The number of updates for weights", "WeightsUpdateCount")]
        public int WeightsUpdateCount = 0;
    }
}
