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
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.ComponentModel.DataAnnotations;
using System.IO;
using TensorSharp.CUDA.ContextState;

namespace Seq2SeqSharp.Applications
{
    public class Options
    {

        [Arg("Automatic mixed precision. Default is false", nameof(AMP))]
        public bool AMP = false;

        [Arg("The batch size", nameof(BatchSize))]
        [Range(1, 99999)]
        public int BatchSize = 1;

        [Arg("The maxmium token size per batch", nameof(MaxTokenSizePerBatch))]
        [Range(1, 999999)]
        public int MaxTokenSizePerBatch = 5000;

        [Arg("Beam search size. Default is 1", nameof(BeamSearchSize))]
        [Range(1, 99)]
        public int BeamSearchSize = 1;

        [Arg("The beta1 for optimizer", nameof(Beta1))]
        [Range(0.5f, 1.0f)]
        public float Beta1 = 0.9f;

        [Arg("The beta2 for optimizer", nameof(Beta2))]
        [Range(0.5f, 1.0f)]
        public float Beta2 = 0.98f;

        [Arg("The options for CUDA NVRTC compiler. Options are split by space. For example: \"--use_fast_math --gpu-architecture=compute_60\"", nameof(CompilerOptions))]
        public string CompilerOptions = "--use_fast_math";

        [Arg("The file path of config file for parameters", nameof(ConfigFilePath))]
        public string ConfigFilePath = string.Empty;

        [Arg("Token generation types. It supports GreedySearch and Sampling. Default is GreedySearch", nameof(DecodingStrategy))]
        [RegularExpression("GreedySearch|Sampling")]
        public DecodingStrategyEnums DecodingStrategy = DecodingStrategyEnums.GreedySearch;

        [Arg("The Top-P value in decoding. Default is 0.0", nameof(DecodingTopP))]
        [Range(0.0f, 1.0f)]
        public float DecodingTopP = 0.0f;

        [Arg("The temperature in decoding, Default value is 1.0f", nameof(DecodingTemperature))]
        [Range(0.0f, 1.0f)]
        public float DecodingTemperature = 1.0f;

        [Arg("The token repeat penalty in decoding, Default value is 2.0f", nameof(DecodingRepeatPenalty))]
        [Range(0.0f, 999.0f)]
        public float DecodingRepeatPenalty = 2.0f;

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", nameof(DeviceIds))]
        public string DeviceIds = "0";

        [Arg("Dropout ratio", nameof(DropoutRatio))]
        [Range(0.0f, 1.0f)]
        public float DropoutRatio = 0.0f;

        [Arg("Enable segment embeddings", nameof(EnableSegmentEmbeddings))]
        public bool EnableSegmentEmbeddings = false;

        [Arg("Enable tensor core. Default is true", nameof(EnableTensorCore))]
        public bool EnableTensorCore = true;

        [Arg("The expert size for MoE model. Default is 1", nameof(ExpertNum))]
        [Range(1, 8192)]
        public int ExpertNum = 1;

        [Arg("It indicates how many experts will be assigned to each token by router in every MoE layer. Default is 1", nameof(ExpertsPerTokenFactor))]
        [Range(1, 8192)]
        public int ExpertsPerTokenFactor = 1;

        [Arg("Maximum Segment Capacity. Default value is 16", nameof(MaxSegmentNum))]
        [Range(1, 8192)]
        public int MaxSegmentNum = 16;

        [Arg("The network depth in encoder.", nameof(EncoderLayerDepth))]
        [Range(1, 8192)]
        public int EncoderLayerDepth = 1;

        [Arg("The embedding dim in source side", nameof(SrcEmbeddingDim))]
        [Range(1, 102400)]
        public int SrcEmbeddingDim = 128;

        [Arg("Starting Learning rate factor for encoders", nameof(EncoderStartLearningRateFactor))]
        [Range(0.0f, 1.0f)]
        public float EncoderStartLearningRateFactor = 1.0f;

        [Arg("Encoder type: None, LSTM, BiLSTM, Transformer", nameof(EncoderType))]
        public EncoderTypeEnums EncoderType = EncoderTypeEnums.Transformer;

        //[Arg("The gamma value of focal loss. Default is 0.0f", nameof(FocalLossGamma))]
        //[Range(0.0f, 5.0f)]
        //public float FocalLossGamma = 0.0f;

        //[Arg("The smooth value of loss. Default is 1e-9f", nameof(LossSmooth))]
        //[Range(1e-12f, 1.0f)]
        //public float LossSmooth = 1e-9f;

        [Arg("Label smoothing. Default is 0.0", nameof(LabelSmoothing))]
        public float LabelSmoothing = 0.0f;

        [Arg("Clip gradients", nameof(GradClip))]
        [Range(0.0000001f, 999.0f)]
        public float GradClip = 3.0f;

        [Arg("The hidden layer size of encoder and decoder.", nameof(HiddenSize))]
        [Range(1, 102400)]
        public int HiddenSize = 128;

        [Arg("The intermediate layer size", nameof(IntermediateSize))]
        [Range(1, 409600)]
        public int IntermediateSize = 512;

        [Arg("The input file for test.", nameof(InputTestFile))]
        public string InputTestFile = null;

        [Arg("It indicates if the encoder is trainable", nameof(IsEncoderTrainable))]
        public bool IsEncoderTrainable = true;

        [Arg("The type of loss function. It supports CrossEntropy and NegativeLogLikelihood", nameof(LossType))]
        [RegularExpression("CrossEntropy|NegativeLogLikelihood")]
        public LossEnums LossType = LossEnums.CrossEntropy;

        [Arg("The maxmium epoch number during training. Default is 100", nameof(MaxEpochNum))]
        [Range(1, 9999)]
        public int MaxEpochNum = 100;

        [Arg("The ratio of memory usage", nameof(MemoryUsageRatio))]
        [Range(0.1f, 1.0f)]
        public float MemoryUsageRatio = 0.95f;

        [Arg("The memory allocator type in Cuda. It supports Basic, CudaMemoryPool and CustomMemoryPool. Default is CudaMemoryPool (Cuda 11.2 or above is required)", nameof(CudaMemoryDeviceAllocatorType))]
        [RegularExpression("Basic|CudaMemoryPool|CustomMemoryPool")]
        public CudaMemoryDeviceAllocatorType CudaMemoryAllocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool;

        [Arg("The trained model file path.", nameof(ModelFilePath))]
        public string ModelFilePath = "Seq2Seq.Model";

        [Arg("The number of multi-heads in transformer model", nameof(MultiHeadNum))]
        [Range(1, 999)]
        public int MultiHeadNum = 8;

        [Arg("The email to notify evaluation result", nameof(NotifyEmail))]
        public string NotifyEmail = "";

        [Arg("The weights optimizer during training. It supports Adam and RMSProp. Adam is default", nameof(Optimizer))]
        [RegularExpression("Adam|RMSProp")]
        public string Optimizer = "Adam";

        [Arg("The test result file.", nameof(OutputFile))]
        public string OutputFile = null;

        [Arg("The prompt for output. It's a input file along with InputTestFile", nameof(OutputPromptFile))]
        public string OutputPromptFile = null;

        [Arg("The processor type: GPU, CPU, CPU_MKL", nameof(ProcessorType))]
        [RegularExpression("GPU|CPU|CPU_MKL")]
        public ProcessorTypeEnums ProcessorType = ProcessorTypeEnums.GPU;

        [Arg("The instructions used in CPU_MKL processor type", nameof(MKLInstructions))]
        [RegularExpression("AVX|AVX2|AVX2_E1|AVX512|AVX512_E1|AVX512_E2|AVX512_E3|AVX512_E4|SSE4_2")]
        public string MKLInstructions = "AVX2";

        [Arg("The source language name.", nameof(SrcLang))]
        public string SrcLang;

        [Arg("The vocabulary file path for source side.", nameof(SrcVocab))]
        public string SrcVocab = null;

        [Arg("Mode to save GPU memory. Default is false", nameof(SaveGPUMemoryMode))]
        public bool SaveGPUMemoryMode = false;

        [Arg("Starting Learning rate", nameof(StartLearningRate))]
        [Range(0.000000001f, 1.0f)]
        public float StartLearningRate = 0.0006f;

        [Arg("The decay steps of learning rate", nameof(LearningRateDecaySteps))]
        [Range(1, 999999999)]
        public int LearningRateDecaySteps = 500000; // 500K

        [Arg("The type of learning rate", nameof(LearningRateType))]
        [RegularExpression("Decay|CosineDecay")]
        public LearningRateTypeEnums LearningRateType = LearningRateTypeEnums.CosineDecay;

        [Arg("The type of token paddings. It could be NoPaddingInSrc, NoPaddingInTgt, NoPadding and AllowPadding. The default value is NoPadding", nameof(PaddingType))]
        [RegularExpression("NoPaddingInSrc|NoPaddingInTgt|NoPadding|AllowPadding")]
        public PaddingEnums PaddingType = PaddingEnums.NoPadding;

        [Arg("The alignment factor when padding sequences. The default value 0 (No alignment)", nameof(PaddingAlignmentFactor))]
        [Range(0, 128)]
        public int PaddingAlignmentFactor = 0;

        [Arg("Task to execute. It supports Train, Valid, Test, DumpVocab, UpdateVocab and Help", nameof(Task))]
        [RegularExpression("Train|Valid|Test|Alignment|DumpVocab|UpdateVocab|VQModel|Help")]
        public ModeEnums Task = ModeEnums.Help;

        [Arg("How to deal with too long sequence. It can be Ignore or Truncation", nameof(TooLongSequence))]
        [RegularExpression("Ignore|Truncation")]
        public TooLongSequence TooLongSequence = TooLongSequence.Ignore;

        [Arg("Activate function used in the model. It can be ReLU, SiLU and LeakyReLU", nameof(ActivateFunc))]
        [RegularExpression("ReLU|SiLU|LeakyReLU")]
        public ActivateFuncEnums ActivateFunc = ActivateFuncEnums.ReLU;

        [Arg("Model vector quantization. Support INT8. Default is disabled.", nameof(VQType))]
        [RegularExpression("None|INT8|INT4|FLOAT16")]
        public VQTypeEnums VQType = VQTypeEnums.None;

        [Arg("The target language name.", nameof(TgtLang))]
        public string TgtLang;

        [Arg("The vocabulary file path for target side.", nameof(TgtVocab))]
        public string TgtVocab = null;

        [Arg("Training corpus folder path", nameof(TrainCorpusPath))]
        public string TrainCorpusPath = null;

        [Arg("Indexed data set file paht. The default value is empty.", nameof(IndexedCorpusPath))]
        public string IndexedCorpusPath = null;

        [Arg("The batch id that the tool will start to process. The default value is 0", nameof(StartBatchId))]
        [Range(0, 9999999)]
        public int StartBatchId = 0;

        [Arg("The max degress of parallelism in task. Default is 1", nameof(TaskParallelism))]
        [Range(1, 999)]
        public int TaskParallelism = 1;

        [Arg("Update parameters every N batches. Default is 1", nameof(UpdateFreq))]
        [Range(1, 99999)]
        public int UpdateFreq = 1;

        [Arg("The maxmium token size per batch during validation", nameof(ValMaxTokenSizePerBatch))]
        [Range(1, 99999)]
        public int ValMaxTokenSizePerBatch = 5000;

        [Arg("Start to run validation after N updates. Default is 20,000", nameof(StartValidAfterUpdates))]
        [Range(1, 9999999)]
        public int StartValidAfterUpdates = 20000;

        [Arg("Run validation every certain updates", nameof(RunValidEveryUpdates))]
        [Range(1, 9999999)]
        public int RunValidEveryUpdates = 10000;


        [Arg("Save checkpoint every certain updates. The default value is 10000", nameof(SaveModelEveryUpdates))]
        [Range(1, 9999999)]
        public int SaveModelEveryUpdates = 10000;

        [Arg("Valid corpus folder path", nameof(ValidCorpusPaths))]
        public string ValidCorpusPaths = null;

        [Arg("The number of steps for warming up", nameof(WarmUpSteps))]
        [Range(1, 9999999)]
        public int WarmUpSteps = 8000;

        [Arg("The number of updates for weights", nameof(WeightsUpdateCount))]
        [Range(1, 9999999)]
        public int WeightsUpdateCount = 0;

        [Arg("The step down factor of learning rate after each epoch. Default is 1.0 which is no step down.", nameof(LearningRateStepDownFactor))]
        [Range(0.01f, 1.0f)]
        public float LearningRateStepDownFactor = 1.0f;
         
        [Arg("The update num to step down learning rate. Default is 0 which means no step down.", nameof(UpdateNumToStepDownLearningRate))]
        [Range(0, 999999)]
        public int UpdateNumToStepDownLearningRate = 0;

        [Arg("The size of vocabulary in source side", nameof(SrcVocabSize))]
        [Range(100, 999999)]
        public int SrcVocabSize = 45000;

        [Arg("The size of vocabulary in target side", nameof(TgtVocabSize))]
        [Range(100, 999999)]
        public int TgtVocabSize = 45000;

        [Arg("It indicates if tags are embedded and applied to normal tokens.", nameof(EnableTagEmbeddings))]
        public bool EnableTagEmbeddings = false;

        [Arg("SentencePiece model for source side", nameof(SrcSentencePieceModelPath))]
        public string SrcSentencePieceModelPath = null;

        [Arg("SentencePiece model for target side", nameof(TgtSentencePieceModelPath))]
        public string TgtSentencePieceModelPath = null;

        [Arg("The minimum token frequency in vocabulary", nameof(MinTokenFreqInVocab))]
        [Range(1, 9999)]
        public int MinTokenFreqInVocab = 1;

        [Arg("The seed value of random generator", nameof(RandomSeed))]
        [Range(-1, 9999999)]
        public int RandomSeed = -1;

        [Arg("Initial loss Scaling when AMP is enabled. Default is 1 which is disabled.", nameof(InitLossScaling))]
        [Range(1, 65000)]
        public float InitLossScaling = 1.0f;

        [Arg("The Positional Embeddings Type. It supports APE, NoPE and RoPE", nameof(PEType))]
        [RegularExpression("APE|NoPE|RoPE")]
        public PositionEmbeddingEnums PEType = PositionEmbeddingEnums.APE;

        [Arg("The type of attention layer. It supports Classic and FlashAttentionV2", nameof(AttentionType))]
        [RegularExpression("Classic|FlashAttentionV2")]
        public AttentionTypeEnums AttentionType = AttentionTypeEnums.Classic;

        [Arg("The type of normalization. It supports LayerNorm and RMSNorm", nameof(NormType))]
        [RegularExpression("LayerNorm|RMSNorm")]
        public NormEnums NormType = NormEnums.RMSNorm;

        [Arg("Log destination. Supported Values: None = 0, Console = 1, LogFile = 2, Callback = 4, and These values can be combined. For example: Value 3 means the log will be outputted to both Console and LogFile", nameof(LogDestination))]
        public Logger.Destination LogDestination = (Logger.Destination.Console | Logger.Destination.Logfile);

        [Arg("The level of logs to be printed out. Supported Values: none = 0, err = 1, warn = 2, info = 4 and debug = 8. These values can be combined. For example: Value 15 means err, warn, info and debug will be outputted.", nameof(LogLevel))]
        public Logger.Level LogLevel = (Logger.Level.err | Logger.Level.warn | Logger.Level.info | Logger.Level.debug);

        [Arg("It indicates if checking tensor corrupted is enabled. Default is disabled.", nameof(CheckTensorCorrupted))]
        public bool CheckTensorCorrupted = false;        

        public void ValidateOptions()
        {
            if (AMP == true && ProcessorType != ProcessorTypeEnums.GPU)
            {
                throw new ArgumentException($"AMP (automatic mixed precesion) is only available for GPUs now. AMP has not supported CPUs yet.");
            }

            if (ProcessorType == ProcessorTypeEnums.GPU && CompilerOptions.Contains("--include-path") == false && AMP == true)
            {
                throw new ArgumentException($"Option --include-path is required in CompilerOptions for GPU tasks. It should points to installed CUDA SDK include path in this machine.");
            }

            // Model must exist if current task is not for training
            if (Task != ModeEnums.Train && !File.Exists(ModelFilePath))
            {
                throw new FileNotFoundException($"Model '{ModelFilePath}' doesn't exist for task '{Task}'");
            }

            if (AttentionType == AttentionTypeEnums.FlashAttentionV2 && ProcessorType != ProcessorTypeEnums.GPU)
            {
                throw new ArgumentException("FlashAttentionV2 runs on GPU only, please use the classic attention layer instead.");
            }
        }
    }
}
