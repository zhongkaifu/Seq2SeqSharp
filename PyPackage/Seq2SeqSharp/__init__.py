from pythonnet import load
load("coreclr")
import clr
import os
import sys

dir = os.path.dirname(sys.modules["Seq2SeqSharp"].__file__)
path = os.path.join(dir, "Seq2SeqSharp.dll")
clr.AddReference(path)

from Seq2SeqSharp.LearningRate import DecayLearningRate
from Seq2SeqSharp.Applications import DecodingOptions, Seq2SeqOptions, GPT, Seq2Seq
from Seq2SeqSharp.Corpus import Seq2SeqCorpusBatch, Seq2SeqCorpus, SeqCorpus
from Seq2SeqSharp.Utils import ProcessorTypeEnums, Misc, ShuffleEnums, Vocab, DecodingStrategyEnums
from Seq2SeqSharp.Enums import ModeEnums, DecoderTypeEnums
from Seq2SeqSharp.Metrics import BleuMetric
from Seq2SeqSharp.Tools import TooLongSequence
from Seq2SeqSharp.Optimizer import AdamOptimizer
