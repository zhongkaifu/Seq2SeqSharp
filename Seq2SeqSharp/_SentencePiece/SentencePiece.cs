using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;
using CC = System.Runtime.InteropServices.CallingConvention;

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
        private static class Native
        {
            static Native() => Init();
            private static bool Isx64() => (IntPtr.Size == 8);
            private static bool IsLinux()
            {
                var p = (int) Environment.OSVersion.Platform;
                return (p == 4) || (p == 6) || (p == 128);
            }

            #region [.DllImport.]
            private const string DLL_WIN_x64 = "sentencepiece.dll"; //"sentencepiece_x64.dll";
            private const string DLL_WIN_x86 = "sentencepiece_x86.dll";
            private const string DLL_LIN_x64 = "libsentencepiece.so";
            private const string DLL_LIN_x86 = DLL_LIN_x64;

            private const string __SP_Init__name     = "__SP_Init";
            private const string __SP_Finalize__name = "__SP_Finalize";
            private const string __SP_Encode__name   = "__SP_Encode";
            private const string __SP_Decode__name   = "__SP_Decode";
            private const string __SP_Free__name     = "__SP_Free";

            public delegate IntPtr __SP_Init_Delegate( string modelFilename, string vocabFilename, int threshold );
            public delegate  void __SP_Finalize_Delegate( IntPtr sp );
            public delegate  IntPtr __SP_Encode_Delegate( IntPtr sp, IntPtr input, int len );
            public delegate  IntPtr __SP_Decode_Delegate( IntPtr sp, IntPtr input, int len );
            public delegate  void __SP_Free_Delegate( IntPtr result );

            #region [.win.]
            #region [.x64.]
            [DllImport(DLL_WIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Init__name)]
            private extern static IntPtr __SP_Init_win_x64( string modelFilename, string vocabFilename, int threshold );

            [DllImport(DLL_WIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Finalize__name)]
            private extern static void __SP_Finalize_win_x64( IntPtr sp );

            [DllImport(DLL_WIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Encode__name)]
            private extern static IntPtr __SP_Encode_win_x64( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_WIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Decode__name)]
            private extern static IntPtr __SP_Decode_win_x64( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_WIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Free__name)]
            private extern static void __SP_Free_win_x64( IntPtr sp );
            #endregion

            #region [.x86.]
            [DllImport(DLL_WIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Init__name)]
            private extern static IntPtr __SP_Init_win_x86( string modelFilename, string vocabFilename, int threshold );

            [DllImport(DLL_WIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Finalize__name)]
            private extern static void __SP_Finalize_win_x86( IntPtr sp );

            [DllImport(DLL_WIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Encode__name)]
            private extern static IntPtr __SP_Encode_win_x86( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_WIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Decode__name)]
            private extern static IntPtr __SP_Decode_win_x86( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_WIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Free__name)]
            private extern static void __SP_Free_win_x86( IntPtr sp );
            #endregion
            #endregion

            #region [.linux.]
            #region [.x64.]
            [DllImport(DLL_LIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Init__name)]
            private extern static IntPtr __SP_Init_lin_x64( string modelFilename, string vocabFilename, int threshold );

            [DllImport(DLL_LIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Finalize__name)]
            private extern static void __SP_Finalize_lin_x64( IntPtr sp );

            [DllImport(DLL_LIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Encode__name)]
            private extern static IntPtr __SP_Encode_lin_x64( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_LIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Decode__name)]
            private extern static IntPtr __SP_Decode_lin_x64( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_LIN_x64, CallingConvention=CC.Cdecl, EntryPoint=__SP_Free__name)]
            private extern static void __SP_Free_lin_x64( IntPtr sp );
            #endregion

            #region [.x86.]
            [DllImport(DLL_LIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Init__name)]
            private extern static IntPtr __SP_Init_lin_x86( string modelFilename, string vocabFilename, int threshold );

            [DllImport(DLL_LIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Finalize__name)]
            private extern static void __SP_Finalize_lin_x86( IntPtr sp );

            [DllImport(DLL_LIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Encode__name)]
            private extern static IntPtr __SP_Encode_lin_x86( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_LIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Decode__name)]
            private extern static IntPtr __SP_Decode_lin_x86( IntPtr sp, IntPtr input, int len );

            [DllImport(DLL_LIN_x86, CallingConvention=CC.Cdecl, EntryPoint=__SP_Free__name)]
            private extern static void __SP_Free_lin_x86( IntPtr sp );
            #endregion
            #endregion
            #endregion

            public static __SP_Init_Delegate     __SP_Init     { [M(O.AggressiveInlining)] get; private set; }
            public static __SP_Finalize_Delegate __SP_Finalize { [M(O.AggressiveInlining)] get; private set; }
            public static __SP_Encode_Delegate   __SP_Encode   { [M(O.AggressiveInlining)] get; private set; }
            public static __SP_Decode_Delegate   __SP_Decode   { [M(O.AggressiveInlining)] get; private set; }
            public static __SP_Free_Delegate     __SP_Free     { [M(O.AggressiveInlining)] get; private set; }

            private static void Init()
            {
                if ( IsLinux() )
                {
                    if ( Isx64() )
                    {
                        __SP_Init     = __SP_Init_lin_x64;
                        __SP_Finalize = __SP_Finalize_lin_x64;
                        __SP_Encode   = __SP_Encode_lin_x64;
                        __SP_Decode   = __SP_Decode_lin_x64;
                        __SP_Free     = __SP_Free_lin_x64;
                    }
                    else
                    {
                        __SP_Init     = __SP_Init_lin_x86;
                        __SP_Finalize = __SP_Finalize_lin_x86;
                        __SP_Encode   = __SP_Encode_lin_x86;
                        __SP_Decode   = __SP_Decode_lin_x86;
                        __SP_Free     = __SP_Free_lin_x86;
                    }
                } 
                else 
                {
                    if ( Isx64() )
                    {
                        __SP_Init     = __SP_Init_win_x64;
                        __SP_Finalize = __SP_Finalize_win_x64;
                        __SP_Encode   = __SP_Encode_win_x64;
                        __SP_Decode   = __SP_Decode_win_x64;
                        __SP_Free     = __SP_Free_win_x64;
                    }
                    else
                    {
                        __SP_Init     = __SP_Init_win_x86;
                        __SP_Finalize = __SP_Finalize_win_x86;
                        __SP_Encode   = __SP_Encode_win_x86;
                        __SP_Decode   = __SP_Decode_win_x86;
                        __SP_Free     = __SP_Free_win_x86;
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public struct InitParams
        {
            public string ModelFilename { get; set; }
            public string VocabFilename { get; set; }
            public int?   Threshold     { get; set; }
        }

        #region [.ctor().]
        private const char SPACE = ' ';

        private IntPtr _SP;
        public SentencePiece( string modelFilename, string vocabFilename = null, int threshold = 1_000 )
        {
            if ( !File.Exists( modelFilename ) ) throw (new FileNotFoundException( null, (modelFilename != null) ? Path.GetFullPath( modelFilename ) : modelFilename ));

            _SP = Native.__SP_Init( modelFilename, vocabFilename, threshold );
        }
        public SentencePiece( in InitParams p ) : this( p.ModelFilename, p.VocabFilename, p.Threshold.GetValueOrDefault( 1_000 ) ) { }
        ~SentencePiece() => Native.__SP_Finalize( _SP );
        public void Dispose()
        {
            Native.__SP_Finalize( _SP );
            GC.SuppressFinalize( this );
        }
        #endregion

        [M(O.AggressiveInlining)] unsafe private static string ConvertToString( IntPtr ptr )
        {
            #region comm. EQUAL/SAME METHOD
            //*
            var result_bytes = (byte*) ptr.ToPointer();
            var len = 0;
            for ( ; ; len++ )
            {
                if ( result_bytes[ len ] == 0 )
                {
                    break;
                }
            }
            var result = Encoding.UTF8.GetString( result_bytes, len );
            //*/
            #endregion

            //var result = Marshal.PtrToStringUTF8( ptr );
            return (result);
        }
        unsafe public string Encode( string input, bool toLower = false )
        {
            if ( toLower )
            {
                input = input.ToLowerInvariant();
            }
            var bytes = Encoding.UTF8.GetBytes( input );
            fixed ( byte* bytes_ptr = bytes )
            {
                var ptr = Native.__SP_Encode( _SP, (IntPtr) bytes_ptr, bytes.Length );

                var result = ConvertToString( ptr );
                Native.__SP_Free( ptr );
                return (result);
            }
        }
        public string Decode( IList< string > words, int startIndex, int count, StringBuilder buff = null )
        {
            if ( buff == null ) buff = new StringBuilder( 0x100 );
            else buff.Clear();

            for ( var end = startIndex + count; startIndex < end; startIndex++ )
            {
                var w = words[ startIndex ];

                if ( buff.Length != 0 ) buff.Append( SPACE );
                buff.Append( w );
            }
            var text = buff.ToString();
            return (Decode( text ));
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
                var ptr = Native.__SP_Decode( _SP, (IntPtr) bytes_ptr, bytes.Length );

                var result = ConvertToString( ptr );
                Native.__SP_Free( ptr );
                return (result/*.Replace( '▁', SPACE ).TrimStart( SPACE )*/);
            }
        }
    }
}
