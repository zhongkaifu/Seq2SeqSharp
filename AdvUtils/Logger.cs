using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AdvUtils
{
    public class Logger
    {
        public enum Level { err, warn, info};
        public enum LogVerbose {None, Normal, Details, Debug };

        public static LogVerbose Verbose = LogVerbose.Normal;

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

            if (level != Level.info)
                Console.Error.WriteLine(sLine);
            else
                Console.WriteLine(sLine);

            if (s_sw != null)
                s_sw.WriteLine(sLine);

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

            Console.ForegroundColor = color;

            if (level != Level.info)
                Console.Error.WriteLine(sLine);
            else
                Console.WriteLine(sLine);

            Console.ResetColor();

            if (s_sw != null)
                s_sw.WriteLine(sLine);

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
