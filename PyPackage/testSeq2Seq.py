from Seq2SeqSharp import Seq2SeqOptions, ModeEnums, ProcessorTypeEnums, DecodingStrategyEnums, Seq2Seq

opts = Seq2SeqOptions()
opts.Task = ModeEnums.Test
opts.ModelFilePath = "../Tests/Seq2SeqSharp.Tests/mt_enu_chs.model"
opts.InputTestFile = "test.src"
opts.OutputFile = "a.out"
opts.ProcessorType = ProcessorTypeEnums.CPU
opts.MaxSrcSentLength = 110
opts.MaxTgtSentLength = 110
opts.BatchSize = 1
opts.SrcSentencePieceModelPath = "./enu.model"
opts.TgtSentencePieceModelPath = "./chs.model"

decodingOptions = opts.CreateDecodingOptions()

ss = Seq2Seq(opts)
ss.Test(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath, "")


