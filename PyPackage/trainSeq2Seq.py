from Seq2SeqSharp import Seq2SeqOptions, ModeEnums, ProcessorTypeEnums, Seq2Seq, Vocab, Seq2SeqCorpus, DecayLearningRate, BleuMetric, Misc, TooLongSequence, ShuffleEnums, AdamOptimizer
import json

def ParseOptions(config_json):
    opts = Seq2SeqOptions()
    opts.Task = ModeEnums.Train
    opts.ProcessorType = ProcessorTypeEnums.CPU
    opts.ModelFilePath = config_json['ModelFilePath']
    opts.RunValidEveryUpdates = int(config_json['RunValidEveryUpdates'])
    opts.UpdateFreq = int(config_json['UpdateFreq'])
    opts.StartValidAfterUpdates = int(config_json['StartValidAfterUpdates'])
    opts.WeightsUpdateCount = int(config_json['WeightsUpdateCount'])
    return opts

with open("train_opts.json", 'r') as file:
    opts = json.load(file)

trainCorpus = Seq2SeqCorpus(corpusFilePath = opts['TrainCorpusPath'], srcLangName = opts['SrcLang'], tgtLangName = opts['TgtLang'], maxTokenSizePerBatch = int(opts['MaxTokenSizePerBatch']), maxSrcSentLength = int(opts['MaxSrcSentLength']), maxTgtSentLength = int(opts['MaxTgtSentLength']))

validCorpusList = []
if len(opts['ValidCorpusPaths']) > 0:
    validCorpusPaths = opts['ValidCorpusPaths'].split(';')
    for validCorpusPath in validCorpusPaths:
        validCorpus = Seq2SeqCorpus(validCorpusPath, opts['SrcLang'], opts['TgtLang'], int(opts['ValMaxTokenSizePerBatch']), int(opts['MaxValidSrcSentLength']), int(opts['MaxValidTgtSentLength']))
        validCorpusList.append(validCorpus)

learningRate = DecayLearningRate(opts['StartLearningRate'], opts['WarmUpSteps'], opts['WeightsUpdateCount'], opts['LearningRateStepDownFactor'], opts['UpdateNumToStepDownLearningRate'])
optimizer = AdamOptimizer(opts['GradClip'], opts['Beta1'], opts['Beta2'], opts['SaveGPUMemoryMode'])

metrics = []
metrics.append(BleuMetric())

srcVocab = Vocab(opts['SrcVocab'])
tgtVocab = Vocab(opts['TgtVocab'])

opts2 = ParseOptions(opts)
decodingOptions = opts2.CreateDecodingOptions()

ss = Seq2Seq(opts2, srcVocab, tgtVocab)
ss.Train(maxTrainingEpoch = opts['MaxEpochNum'], trainCorpus = trainCorpus, validCorpusList = validCorpusList, learningRate = learningRate, optimizer = optimizer, metrics = metrics, decodingOptions = decodingOptions);


