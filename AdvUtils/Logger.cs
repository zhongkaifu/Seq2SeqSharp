using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace AdvUtils
{
    /// <summary>
    /// Progress Callback delegate with three functionalities: 
    ///   1. post a callback message to the caller routine 
    ///   2. post a callback progress value (%) to the caller routine for long operations 
    ///   3. Signal if the long process must be canceled for stopping long operations on request of the caller
    /// </summary>
    /// <param name="value">progress value in % (0-100)</param>
    /// <param name="log">progress message</param>
    /// <param name="type">type of message, for example, 0: log, 1: error, etc.</param>
    /// <param name="color">request a specific color for the message</param>
    /// <returns>+1 if the process should be canceled, -1 if not</returns>
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    public delegate int ProgressCallback(
        int value,
        StringBuilder log,
        int type,
        int color = 0
        );

    public class Logger
    {
        public enum Level { err, warn, info};

        public enum LogVerbose {None, Normal, Details, Debug, Callback, CallbackDetails, Logfileonly, Progress };

        public static LogVerbose Verbose = LogVerbose.Normal;

        private static ProgressCallback? s_callback = null;

        private static LogVerbose s_logverbosebackup = LogVerbose.Normal;

        /// <summary>
        /// Set the callback routine in your code with this to automatically redirect all messages to your callback function
        /// </summary>
        public static ProgressCallback? Callback
        {
            get => s_callback;
            set
            {
                s_callback = value;
                if (s_callback != null)
                {
                    s_logverbosebackup = Verbose;
                    Verbose = LogVerbose.Callback;
                }
                else
                {
                    Verbose = s_logverbosebackup;
                }
            }
        }

        public static void WriteLine(string s, params object[] args)
        {
            if (Verbose == LogVerbose.None)
            {
                return;
            }

            WriteLine(Level.info, s, args);
        }

        public static void WriteLine(Level level, string s, params object[] args)
        {
            if (Verbose == LogVerbose.None)
            {
                return;
            }

            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("{0},{1} ", level.ToString(), DateTime.Now.ToString());

            if (args.Length == 0)
                sb.Append(s);
            else
                sb.AppendFormat(s, args);

            string sLine = sb.ToString();

            if (Callback != null && (Verbose == LogVerbose.Callback || Verbose == LogVerbose.CallbackDetails))
            { // let the caller handle the message
                StringBuilder sbl = new StringBuilder(sLine);
                Callback(0, sbl, (int)level);
            }
            else if (Callback != null && Verbose == LogVerbose.Progress)
            { // inform the caller about the progress
                if (args.Length > 0)
                {
                    StringBuilder sbl0 = new StringBuilder("");
                    Callback((int)args[0], sbl0, (int)level);
                    return;
                }
            }
            else if (Verbose != LogVerbose.Logfileonly)
            { // only print on the Console if Logfileonly is not requested
                if (level != Level.info)
                    Console.Error.WriteLine(sLine);
                else
                    Console.WriteLine(sLine);
            }

            try
            {
                if (s_sw != null)
                    s_sw.WriteLine(sLine);
            }
            catch (Exception err)
            {
                if (Callback != null && (Verbose == LogVerbose.Callback || Verbose == LogVerbose.CallbackDetails))
                { // let the caller handle the message
                    StringBuilder sbl = new StringBuilder($"Failed to write to log file '{LogFile}'. Error = '{err.Message}'");
                    Callback(0, sbl, (int)level);
                }
                else
                {
                    Console.Error.WriteLine($"Failed to write to log file '{LogFile}'. Error = '{err.Message}'");
                }
                s_sw = null;
            }
        }

        public static void WriteLine(Level level, ConsoleColor color, string s, params object[] args)
        {
            if (Verbose == LogVerbose.None)
            {
                return;
            }

            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("{0},{1} ", level.ToString(), DateTime.Now.ToString());

            if (args.Length == 0)
                sb.Append(s);
            else
                sb.AppendFormat(s, args);

            string sLine = sb.ToString();

            if (Callback != null && (Verbose == LogVerbose.Callback || Verbose == LogVerbose.CallbackDetails))
            { // let the caller handle the message
                StringBuilder sbl = new StringBuilder(sLine);
                Callback(0, sbl, (int)level, (int)color);
            }
            else if (Callback != null && Verbose == LogVerbose.Progress)
            { // inform the caller about the progress
                StringBuilder sbl0 = new StringBuilder("");
                Callback((int)args[0], sbl0, (int)level);
                return;
            }
            else if (Verbose != LogVerbose.Logfileonly)
            { // only print on the Console if Logfileonly is not requested
                Console.ForegroundColor = color;

                if (level != Level.info)
                    Console.Error.WriteLine(sLine);
                else
                    Console.WriteLine(sLine);

                Console.ResetColor();
            }

            try
            {
                if (s_sw != null)
                    s_sw.WriteLine(sLine);
            }
            catch (Exception err)
            {
                if (Callback != null && (Verbose == LogVerbose.Callback || Verbose == LogVerbose.CallbackDetails))
                { // let the caller handle the message
                    StringBuilder sbl = new StringBuilder($"Failed to write to log file '{LogFile}'. Error = '{err.Message}'");
                    Callback(0, sbl, (int)level);
                }
                else
                {
                    Console.Error.WriteLine($"Failed to write to log file '{LogFile}'. Error = '{err.Message}'");
                }

                s_sw = null;
            }
        }

        public static void Close()
        {
            if (s_sw != null)
            {
                s_sw.Close();
                s_sw = null;
            }

        }

        public static string? LogFile
        {
            get => s_strLogfile;
            set
            {
                if (s_strLogfile == value)
                    return;

                if (s_sw != null)
                {
                    s_sw.Close();
                    s_sw = null;
                }

                s_strLogfile = value;
                if (s_strLogfile != null)
                {
                    s_sw = new StreamWriter(s_strLogfile, true, Encoding.UTF8);
                    s_sw.AutoFlush = true;
                }
            }
        }

        private static string? s_strLogfile = null;
        private static StreamWriter? s_sw = null;
    }
}
