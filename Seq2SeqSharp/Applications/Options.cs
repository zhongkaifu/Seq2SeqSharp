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
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Applications
{
    public class Options
    {
        [Arg("The batch size", nameof(BatchSize))]
        public int BatchSize = 1;

        [Arg("The maxmium token size per batch", nameof(MaxTokenSizePerBatch))]
        public int MaxTokenSizePerBatch = 5000;

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
        public DecodingStrategyEnums DecodingStrategy = DecodingStrategyEnums.GreedySearch;

        [Arg("The penalty for decoded repeat tokens. Default is 5.0", nameof(DecodingRepeatPenalty))]
        public float DecodingRepeatPenalty = 5.0f;

        [Arg("Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2", nameof(DeviceIds))]
        public string DeviceIds = "0";

        [Arg("Dropout ratio", nameof(DropoutRatio))]
        public float DropoutRatio = 0.0f;

        [Arg("Enable segment embeddings", nameof(EnableSegmentEmbeddings))]
        public bool EnableSegmentEmbeddings = false;

        [Arg("The expert size for MoE model. Default is 1", nameof(ExpertNum))]
        public int ExpertNum = 1;

        [Arg("The factor of expert size per token for MoE model. Default is 1", nameof(ExpertsPerTokenFactor))]
        public int ExpertsPerTokenFactor = 1;

        [Arg("Maximum Segment Capacity. Default value is 16", nameof(MaxSegmentNum))]
        public int MaxSegmentNum = 16;

        [Arg("The network depth in encoder.", nameof(EncoderLayerDepth))]
        public int EncoderLayerDepth = 1;

        [Arg("The embedding dim in source side", nameof(SrcEmbeddingDim))]
        public int SrcEmbeddingDim = 128;

        [Arg("Starting Learning rate factor for encoders", nameof(EncoderStartLearningRateFactor))]
        public float EncoderStartLearningRateFactor = 1.0f;

        [Arg("Encoder type: None, LSTM, BiLSTM, Transformer", nameof(EncoderType))]
        public EncoderTypeEnums EncoderType = EncoderTypeEnums.Transformer;

        [Arg("The gamma value of focal loss. Default is 0.0f", nameof(FocalLossGamma))]
        public float FocalLossGamma = 0.0f;

        [Arg("The smooth value of loss. Default is 1e-9f", nameof(LossSmooth))]
        public float LossSmooth = 1e-9f;

        [Arg("Clip gradients", nameof(GradClip))]
        public float GradClip = 3.0f;

        [Arg("The hidden layer size of encoder and decoder.", nameof(HiddenSize))]
        public int HiddenSize = 128;

        [Arg("The input file for test.", nameof(InputTestFile))]
        public string InputTestFile = null;

        [Arg("It indicates if the encoder is trainable", nameof(IsEncoderTrainable))]
        public bool IsEncoderTrainable = true;

        [Arg("The type of loss function. It supports CrossEntropy and NegativeLogLikelihood", nameof(LossType))]
        public LossEnums LossType = LossEnums.CrossEntropy;

        [Arg("The maxmium epoch number during training. Default is 100", nameof(MaxEpochNum))]
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

        [Arg("The prompt for output. It's a input file along with InputTestFile", nameof(OutputPromptFile))]
        public string OutputPromptFile = null;

        [Arg("The processor type: GPU, CPU, CPU_MKL", nameof(ProcessorType))]
        public ProcessorTypeEnums ProcessorType = ProcessorTypeEnums.GPU;

        [Arg("The instructions used in CPU_MKL processor type", nameof(MKLInstructions))]
        public string MKLInstructions = "AVX2";

        [Arg("The source language name.", nameof(SrcLang))]
        public string SrcLang;

        [Arg("The vocabulary file path for source side.", nameof(SrcVocab))]
        public string SrcVocab = null;

        [Arg("Starting Learning rate", nameof(StartLearningRate))]
        public float StartLearningRate = 0.0006f;

        [Arg("Shuffle Type. It could be NoPaddingInSrc, NoPaddingInTgt and Random", nameof(ShuffleType))]
        public ShuffleEnums ShuffleType = ShuffleEnums.Random;

        [Arg("Task to execute. It supports Train, Valid, Test, DumpVocab, UpdateVocab and Help", nameof(Task))]
        public ModeEnums Task = ModeEnums.Help;

        [Arg("How to deal with too long sequence. It can be Ignore or Truncation", nameof(TooLongSequence))]
        public TooLongSequence TooLongSequence = TooLongSequence.Ignore;

        [Arg("Activate function used in the model. It can be Relu or Swish", nameof(ActivateFunc))]
        public ActivateFuncEnums ActivateFunc = ActivateFuncEnums.Relu;

        [Arg("The level of log to output", nameof(LogVerbose))]
        public Logger.LogVerbose LogVerbose = Logger.LogVerbose.Normal;

        [Arg("The target language name.", nameof(TgtLang))]
        public string TgtLang;

        [Arg("The vocabulary file path for target side.", nameof(TgtVocab))]
        public string TgtVocab = null;

        [Arg("Training corpus folder path", nameof(TrainCorpusPath))]
        public string TrainCorpusPath = null;

        [Arg("The max degress of parallelism in task. Default is 1", nameof(TaskParallelism))]
        public int TaskParallelism = 1;

        [Arg("Update parameters every N batches. Default is 1", nameof(UpdateFreq))]
        public int UpdateFreq = 1;

        [Arg("The maxmium token size per batch during validation", nameof(ValMaxTokenSizePerBatch))]
        public int ValMaxTokenSizePerBatch = 5000;

        [Arg("Start to run validation after N updates. Default is 20,000", nameof(StartValidAfterUpdates))]
        public int StartValidAfterUpdates = 20000;

        [Arg("Run validation every certain updates", nameof(RunValidEveryUpdates))]
        public int RunValidEveryUpdates = 10000;

        [Arg("Valid corpus folder path", nameof(ValidCorpusPaths))]
        public string ValidCorpusPaths = null;

        [Arg("The number of steps for warming up", nameof(WarmUpSteps))]
        public int WarmUpSteps = 8000;

        [Arg("The number of updates for weights", nameof(WeightsUpdateCount))]
        public int WeightsUpdateCount = 0;

        [Arg("The step down factor of learning rate after each epoch. Default is 1.0 which is no step down.", nameof(LearningRateStepDownFactor))]
        public float LearningRateStepDownFactor = 1.0f;

        [Arg("The update num to step down learning rate. Default is 0 which means no step down.", nameof(UpdateNumToStepDownLearningRate))]
        public int UpdateNumToStepDownLearningRate = 0;

        [Arg("The size of vocabulary in source side", nameof(SrcVocabSize))]
        public int SrcVocabSize = 45000;

        [Arg("The size of vocabulary in target side", nameof(TgtVocabSize))]
        public int TgtVocabSize = 45000;

        [Arg("It indicates if tags are embedded and applied to normal tokens.", nameof(EnableTagEmbeddings))]
        public bool EnableTagEmbeddings = false;

        [Arg("SentencePiece model for source side", nameof(SrcSentencePieceModelPath))]
        public string SrcSentencePieceModelPath = null;

        [Arg("SentencePiece model for target side", nameof(TgtSentencePieceModelPath))]
        public string TgtSentencePieceModelPath = null;

        [Arg("The minimum token frequency in vocabulary", nameof(MinTokenFreqInVocab))]
        public int MinTokenFreqInVocab = 1;
    }
}
