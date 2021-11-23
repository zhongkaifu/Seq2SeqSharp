using System.Collections.Generic;
using System.Linq;

using Seq2SeqSharp;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Corpus;

namespace Seq2SeqWebAPI
{
    public static class Seq2SeqInstance
    {
        static private object locker = new object();
        static private Seq2Seq m_seq2seq;
        private static SentencePiece _SrcSentPiece;
        private static SentencePiece _TgtSentPiece;

        static public void Initialization( string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength,
            ProcessorTypeEnums processorType, string deviceIds, in (SentencePiece src, SentencePiece tgt) sentPieces )
        {
            var opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSrcSentLength = maxTestSrcSentLength;
            opts.MaxTestTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            _SrcSentPiece = sentPieces.src;
            _TgtSentPiece = sentPieces.tgt;

            m_seq2seq = new Seq2Seq( opts );
        }

        static public string Call( string input )
        {
            if ( string.IsNullOrWhiteSpace( input ) ) return (input);

            input = _SrcSentPiece.Encode( input );

            var tokens = input.Split( ' ' ).ToList();
            var batchTokens = new List<List<string>> { tokens };
            var groupBatchTokens = new List<List<List<string>>> { batchTokens };

            lock ( locker )
            {
                var nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>( groupBatchTokens );
                var out_tokens = nrs[ 0 ].Output[ 0 ][ 0 ];
                var rst = _TgtSentPiece.Decode( out_tokens, 1, out_tokens.Count - 2 );
                //---var rst = string.Join( " ", out_tokens.ToArray(), 1, out_tokens.Count - 2 );
                return (rst);
            }
        }
    }
}
