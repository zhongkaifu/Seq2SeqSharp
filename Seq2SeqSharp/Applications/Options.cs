using AdvUtils;

namespace Seq2SeqSharp.Applications
{
    public class Options
    {
        [Arg("The batch size", nameof(BatchSize))]
        public int BatchSize = 1;

        [Arg("Beam search size. Default is 1", nameof(BeamSearchSize))]
        public int BeamSearchSize = 1;

        [Arg("The beta1 for optimizer", nameof(Beta1))]
        public float Beta1 = 0.9f;

        [Arg("The beta2 for optimizer", nameof(Beta2))]
        public float Beta2 = 0.98f;

        [Arg("The options for CUDA NVRTC compiler. Options are split by space. For example: \"--use_fast_math --gpu-architecture=compute_60\"", nameof(CompilerOptions))]
        public string CompilerOptions = "--use_fast_math";

        [Arg("The file path of config file for parameters", nameof(ConfigFilePath))]
        public string ConfigFilePath = string.Empty;

        [Arg("Token generation types. It supports GreedySearch and Sampling. Default is GreedySearch", nameof(DecodingStrategy))]
        public string DecodingStrategy = "GreedySearch";

        [Arg("The top-P value for sampling decoding strategy. The value above 0.0 will cause non-deterministic results. Default is 0.0", nameof(DecodingTopPValue))]
        public float DecodingTopPValue = 0.0f;


        [Arg("The penalty for decoded repeat tokens. Default is 2.0", nameof(DecodingRepeatPenalty))]
        public float DecodingRepeatPenalty = 2.0f;

        [Arg("The penalty for decoded token distance. Default is 10.0", nameof(DecodingDistancePenalty))]
        public float DecodingDistancePenalty = 5.0f;

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", nameof(DeviceIds))]
        public string DeviceIds = "0";

        [Arg("Dropout ratio", nameof(DropoutRatio))]
        public float DropoutRatio = 0.0f;

        [Arg("Enable segment embeddings", nameof(EnableSegmentEmbeddings))]
        public bool EnableSegmentEmbeddings = false;

        [Arg("Maximum Segment Capacity. Default value is 16", nameof(MaxSegmentNum))]
        public int MaxSegmentNum = 16;

        [Arg("The network depth in encoder.", nameof(EncoderLayerDepth))]
        public int EncoderLayerDepth = 1;

        [Arg("Starting Learning rate factor for encoders", nameof(EncoderStartLearningRateFactor))]
        public float EncoderStartLearningRateFactor = 1.0f;

        [Arg("Encoder type: LSTM, BiLSTM, Transformer", nameof(EncoderType))]
        public string EncoderType = "Transformer";

        [Arg("Clip gradients", nameof(GradClip))]
        public float GradClip = 3.0f;

        [Arg("The hidden layer size of encoder and decoder.", nameof(HiddenSize))]
        public int HiddenSize = 128;

        [Arg("The input file for test.", nameof(InputTestFile))]
        public string InputTestFile = null;

        [Arg("It indicates if the encoder is trainable", nameof(IsEncoderTrainable))]
        public bool IsEncoderTrainable = true;

        [Arg("Maxmium epoch number during training. Default is 100", nameof(MaxEpochNum))]
        public int MaxEpochNum = 100;

        [Arg("The ratio of memory usage", nameof(MemoryUsageRatio))]
        public float MemoryUsageRatio = 0.95f;

        [Arg("The trained model file path.", nameof(ModelFilePath))]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The number of multi-heads in transformer model", nameof(MultiHeadNum))]
        public int MultiHeadNum = 8;

        [Arg("The email to notify evaluation result", nameof(NotifyEmail))]
        public string NotifyEmail = "";

        [Arg("The weights optimizer during training. It supports Adam and RMSProp. Adam is default", nameof(Optimizer))]
        public string Optimizer = "Adam";

        [Arg("The test result file.", nameof(OutputFile))]
        public string OutputFile = null;

        [Arg("Processor type: GPU, CPU", nameof(ProcessorType))]
        public string ProcessorType = "GPU";

        [Arg("Source language name.", nameof(SrcLang))]
        public string SrcLang;

        [Arg("The vocabulary file path for source side.", nameof(SrcVocab))]
        public string SrcVocab = null;

        [Arg("Starting Learning rate", nameof(StartLearningRate))]
        public float StartLearningRate = 0.0006f;

        [Arg("The shuffle block size", nameof(ShuffleBlockSize))]
        public int ShuffleBlockSize = -1;

        [Arg("Shuffle Type. It could be NoPaddingInSrc, NoPaddingInTgt and Random", nameof(ShuffleType))]
        public string ShuffleType = "Random";

        [Arg("Task to execute. It could be Train, Valid, Test, DumpVocab or Help", nameof(Task), false)]
        public string Task = "Help";

        [Arg("How to deal with too long sequence. It can be Ignore or Truncation", nameof(TooLongSequence))]
        public string TooLongSequence = "Ignore";

        [Arg("Target language name.", nameof(TgtLang))]
        public string TgtLang;

        [Arg("The vocabulary file path for target side.", nameof(TgtVocab))]
        public string TgtVocab = null;

        [Arg("Training corpus folder path", nameof(TrainCorpusPath))]
        public string TrainCorpusPath = null;

        [Arg("Update parameters every N batches. Default is 1", nameof(UpdateFreq))]
        public int UpdateFreq = 1;

        [Arg("The batch size during validation", nameof(ValBatchSize))]
        public int ValBatchSize = 1;

        [Arg("Valid corpus folder path", nameof(ValidCorpusPaths))]
        public string ValidCorpusPaths = null;

        [Arg("The number of steps for warming up", nameof(WarmUpSteps))]
        public int WarmUpSteps = 8000;

        [Arg("The number of updates for weights", nameof(WeightsUpdateCount))]
        public int WeightsUpdateCount = 0;

        [Arg("The interval hours to run model validation", nameof(ValidIntervalHours))]
        public float ValidIntervalHours = 1.0f;

        [Arg("The size of vocabulary in source side", nameof(SrcVocabSize))]
        public int SrcVocabSize = 45000;

        [Arg("The size of vocabulary in target side", nameof(TgtVocabSize))]
        public int TgtVocabSize = 45000;

        [Arg("It indicates if contextual features' embeddings are applied to entire input sequence rather than the first segment", nameof(ApplyContextEmbeddingsToEntireSequence))]
        public bool ApplyContextEmbeddingsToEntireSequence = true;
    }
}
