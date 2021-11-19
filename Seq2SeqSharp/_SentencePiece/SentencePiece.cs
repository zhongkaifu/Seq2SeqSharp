using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Seq2SeqSharp._SentencePiece
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class SentencePiece : IDisposable
    {
        /// <summary>
        /// 
        /// </summary>
        public struct InitParams
        {
            public string ModelFilename { get; set; }
            public string VocabFilename { get; set; }
            public int?   Threshold     { get; set; }
        }

        #region [.DllImport.]
        private const string DLL_SP = @"sentencepiece.dll";
        [DllImport(DLL_SP, CallingConvention=CallingConvention.Cdecl)] private extern static IntPtr __SP_Init( string modelFilename, string vocabFilename, int threshold );
        [DllImport(DLL_SP, CallingConvention=CallingConvention.Cdecl)] private extern static void __SP_Finalize( IntPtr sp );
        [DllImport(DLL_SP, CallingConvention=CallingConvention.Cdecl)] private extern static IntPtr __SP_Encode( IntPtr sp, IntPtr input, int len );
        [DllImport(DLL_SP, CallingConvention=CallingConvention.Cdecl)] private extern static IntPtr __SP_Decode( IntPtr sp, IntPtr input, int len );
        [DllImport(DLL_SP, CallingConvention=CallingConvention.Cdecl)] private extern static void __SP_Free( IntPtr result );
        #endregion

        #region [.ctor().]
        private const char SPACE = ' ';

        private IntPtr _SP;
        public SentencePiece( string modelFilename, string vocabFilename = null, int threshold = 1_000 )
        {
            if ( !File.Exists( modelFilename ) ) throw (new FileNotFoundException( null, (modelFilename != null) ? Path.GetFullPath( modelFilename ) : modelFilename ));

            _SP = __SP_Init( modelFilename, vocabFilename, threshold );
        }
        public SentencePiece( in InitParams p ) : this( p.ModelFilename, p.VocabFilename, p.Threshold.GetValueOrDefault( 1_000 ) ) { }
        ~SentencePiece() => __SP_Finalize( _SP );
        public void Dispose()
        {
            __SP_Finalize( _SP );
            GC.SuppressFinalize( this );
        }
        #endregion

        unsafe public string Encode( string input, bool toLower = true )
        {
            if ( toLower )
            {
                input = input.ToLowerInvariant();
            }
            var bytes = Encoding.UTF8.GetBytes( input );
            fixed ( byte* bytes_ptr = bytes )
            {
                var ptr = __SP_Encode( _SP, (IntPtr) bytes_ptr, bytes.Length );

                #region comm. EQUAL/SAME METHOD
                /*
                var result_bytes = (byte*) ptr.ToPointer();
                var len = 0;
                for ( ; ; len++ )
                {
                    if ( result_bytes[ len ] == 0 )
                    {
                        break;
                    }
                }
                var s = Encoding.UTF8.GetString( result_bytes, len );
                //*/
                #endregion

                var result = Marshal.PtrToStringUTF8( ptr );
                __SP_Free( ptr );
                return (result);
            }
        }
        public string Decode( IList< string > words, StringBuilder buff = null ) //=> Decode( string.Join( " ", words ) );
        {
            if ( buff == null ) buff = new StringBuilder( 0x100 ); 
            else buff.Clear();

            foreach ( var w in words )
            {
                if ( buff.Length != 0 ) buff.Append( SPACE );
                buff.Append( w );
            }
            var text = buff.ToString();
            return (Decode( text ));
        }
        unsafe public string Decode( string input )
        {
            var bytes = Encoding.UTF8.GetBytes( input );
            fixed ( byte* bytes_ptr = bytes )
            {
                var ptr = __SP_Decode( _SP, (IntPtr) bytes_ptr, bytes.Length );

                var result = Marshal.PtrToStringUTF8( ptr );
                __SP_Free( ptr );
                return (result.Replace( '▁', SPACE ).TrimStart( SPACE ));
            }
        }
    }
}
