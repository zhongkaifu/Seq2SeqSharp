using System;
using System.Collections.Generic;
using System.Linq;
using AdvUtils;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;

namespace Seq2SeqWebAPI
{
    public static class Seq2SeqInstance
    {
        static private object locker = new object();
        static private Seq2Seq m_seq2seq;
        private static SentencePiece _SrcSentPiece;
        private static SentencePiece _TgtSentPiece;
        private static Seq2SeqOptions opts;

        static public void Initialization( string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength,
            ProcessorTypeEnums processorType, string deviceIds, in (SentencePiece src, SentencePiece tgt) sentPieces )
        {
            opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxSrcSentLength = maxTestSrcSentLength;
            opts.MaxTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            _SrcSentPiece = sentPieces.src;
            _TgtSentPiece = sentPieces.tgt;


            Logger.WriteLine($"Loading model from '{modelFilePath}', MaxTestSrcSentLength = '{maxTestSrcSentLength}', MaxTestTgtSentLength = '{maxTestTgtSentLength}', ProcessorType = '{opts.ProcessorType}'");

            m_seq2seq = new Seq2Seq( opts );
        }

        static public string Call(string input)
        {
            if (string.IsNullOrWhiteSpace(input)) return (input);

            if (_SrcSentPiece != null)
            {
                input = _SrcSentPiece.Encode(input);
            }

            var tokens = input.Split(' ').ToList();
            var batchTokens = new List<List<string>> { tokens };

            DecodingOptions decodingOptions = opts.CreateDecodingOptions();
            var nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>(batchTokens, null, decodingOptions);
            var out_tokens = nrs[0].Output[0][0];

            string rst = null;
            if (_TgtSentPiece != null)
            {
                rst = _TgtSentPiece.Decode(out_tokens, 1, out_tokens.Count - 2);
            }
            else
            {
                rst = String.Join(" ", out_tokens);
            }

            return rst;
        }
    }
}
