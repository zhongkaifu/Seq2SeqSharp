using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using static AdvUtils.Logger;
using System.Drawing;
using System.Runtime.InteropServices;

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

    public static class Logger
    {
        public enum Destination // Where logs will be going to
        {
            None = 0,
            Console = 1,
            Logfile = 2,
            Callback = 4
        };

        public enum Level
        {
            none = 0,
            err = 1,
            warn = 2,
            info = 4,
            debug = 8,
        };

        private static Level m_logLevel;
        private static ProgressCallback? s_callback = null;

        /// <summary>
        /// Set the callback routine in your code with this to automatically redirect all messages to your callback function
        /// </summary>
        public static ProgressCallback? Callback
        {
            get => s_callback;
            set
            {
                s_callback = value;
            }
        }

        static public void Initialize(Destination dest, Level logLevel, string logFilePath = "", ProgressCallback callback = null)
        {
            if ((dest & Destination.Console) == Destination.Console)
            {
                TextWriterTraceListener tr = new TextWriterTraceListener(System.Console.Out);
                Trace.Listeners.Add(tr);
            }

            if ((dest & Destination.Logfile) == Destination.Logfile)
            {
                TextWriterTraceListener tr = new TextWriterTraceListener(System.IO.File.CreateText(logFilePath));
                Trace.Listeners.Add(tr);
            }

            m_logLevel = logLevel;
            s_callback = callback;
        }

        public static void WriteLine(string s, params object[] args)
        {
            WriteLine(Level.info, ConsoleColor.White, s, args);
        }

        public static void WriteLine(Level level, string s, params object[] args)
        {
            WriteLine(level, ConsoleColor.White, s, args);
        }

        public static void WriteLine(Level level, ConsoleColor color, string s, params object[] args)
        {
            if ((level & m_logLevel) == level)
            {
                Console.ForegroundColor = color;

                StringBuilder sb = new StringBuilder();
                sb.AppendFormat("{0},{1} ", level.ToString(), DateTime.Now.ToString());

                if (args.Length == 0)
                    sb.Append(s);
                else
                    sb.AppendFormat(s, args);

                string sLine = sb.ToString();
                Trace.WriteLine(sLine);
                Trace.Flush();

                if (Callback != null)
                {
                    int progress = 0;
                    if(args.Length > 0) 
                    {
                        int.TryParse((string)args[0], out progress);
                    }
                    Callback(progress, sb, (int)level, (int)color);
                }
            }

        }
    }
}
