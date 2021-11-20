using System;
using System.Diagnostics;
using Seq2SeqSharp._SentencePiece;

namespace libsentencepiece_testapp
{
    internal static class Program
    {
        private static void Test( string modelFilename, string origin_text )
        {
            using ( var sp = new SentencePiece( modelFilename ) )
            {
                Console.WriteLine( $"\r\norigin_text: '{origin_text}'" );

                var encoded_text = sp.Encode( origin_text ); Console.WriteLine( $"encoded_text: '{encoded_text}'" );
                var decoded_text = sp.Decode( encoded_text ); Console.WriteLine( $"decoded_text: '{decoded_text}'" );

                Debug.Assert( origin_text == decoded_text );
            }
        }

        private static void Main( string[] args )
        {
            try
            {
                Test( "../spm/rusSpm.model", origin_text: "бабушка козлика очень любила" );
                Test( "../spm/enuSpm.model", origin_text: "grandmother loved the goat very much" );
            }
            catch ( Exception ex )
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine( ex );
                Console.ResetColor();
            }
            //Console.ReadLine();
        }
    }
}
