from Seq2SeqSharp import Seq2SeqOptions, ModeEnums, ProcessorTypeEnums, GPT, Vocab, SeqCorpus, DecayLearningRate, BleuMetric, Misc, TooLongSequence, ShuffleEnums, AdamOptimizer
from Seq2SeqSharp import DecoderTypeEnums
import json

def ParseOptions(config_json):
    opts = Seq2SeqOptions()
    opts.Task = ModeEnums.Train
    opts.ShuffleType = ShuffleEnums.Random
    opts.ProcessorType = ProcessorTypeEnums.GPU
    opts.DecoderType = DecoderTypeEnums.GPTDecoder
    opts.ModelFilePath = "gpt_test.model"
    opts.RunValidEveryUpdates = int(config_json['RunValidEveryUpdates'])
    opts.UpdateFreq = int(config_json['UpdateFreq'])
    opts.StartValidAfterUpdates = int(config_json['StartValidAfterUpdates'])
    opts.WeightsUpdateCount = int(config_json['WeightsUpdateCount'])

    return opts


#time.sleep(30)


with open("train_opts.json", 'r') as file:
    opts = json.load(file)

trainCorpus = SeqCorpus(corpusFilePath = opts['TrainCorpusPath'], tgtLangName = opts['TgtLang'], maxTokenSizePerBatch = int(opts['MaxTokenSizePerBatch']), maxTgtSentLength = int(opts['MaxTgtSentLength'])) #, shuffleEnums = ShuffleEnums(opts['ShuffleType']), tooLongSequence = TooLongSequence(opts['TooLongSequence']));

learningRate = DecayLearningRate(opts['StartLearningRate'], opts['WarmUpSteps'], opts['WeightsUpdateCount'], opts['LearningRateStepDownFactor'], opts['UpdateNumToStepDownLearningRate'])
optimizer = AdamOptimizer(opts['GradClip'], opts['Beta1'], opts['Beta2'], opts['SaveGPUMemoryMode'])
tgtVocab = Vocab(opts['TgtVocab'])

opts2 = ParseOptions(opts)
decodingOptions = opts2.CreateDecodingOptions()

ss = GPT(opts2, tgtVocab)

ss.Train(maxTrainingEpoch = opts['MaxEpochNum'], trainCorpus = trainCorpus, validCorpusList = None, learningRate = learningRate, optimizer = optimizer, metrics = None, decodingOptions = decodingOptions);


