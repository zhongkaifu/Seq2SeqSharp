from Seq2SeqSharp import Seq2SeqOptions, ModeEnums, ProcessorTypeEnums, DecodingStrategyEnums, GPT

opts = Seq2SeqOptions()
opts.Task = ModeEnums.Test
opts.ModelFilePath = "./ybook_base_v7.model"
opts.InputTestFile = "input.txt"
opts.OutputFile = "a.out"
opts.ProcessorType = ProcessorTypeEnums.CPU
opts.DecodingStrategy = DecodingStrategyEnums.Sampling
opts.MaxSrcSentLength = 110
opts.MaxTgtSentLength = 110
opts.BatchSize = 1
opts.DeviceIds = "0"
opts.DecodingTopP = 0.2
opts.DecodingTemperature = 1.0
opts.SrcSentencePieceModelPath = "./chsYBSpm.model"
opts.TgtSentencePieceModelPath = "./chsYBSpm.model"

decodingOptions = opts.CreateDecodingOptions()

ss = GPT(opts)
ss.Test(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, "")


