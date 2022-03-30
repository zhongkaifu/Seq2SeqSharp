using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using AdvUtils;

namespace SeqDictMatchConsole
{
    class Program
    {
        public static void DictMatchSequences(string inputFilePath, string outputFilePath, string dictFilePath)
        {
            Console.WriteLine("Load raw text dictionary...");
            DictMatch match = new DictMatch();
            match.LoadDictFromRawText(dictFilePath);

            Console.WriteLine("Verify raw text dictionary...");
            Match(inputFilePath, outputFilePath, match);
        }


        //Read each line from strTextFileName, and verify wether terms in every line are in strDictFileName
        public static void Match(string inputFilePath, string outputFilePath, DictMatch match)
        {
            List<Lemma> dm_r = new List<Lemma>();
            List<int> offsetList = new List<int>();

            StreamReader sr = new StreamReader(inputFilePath);
            StreamWriter sw = new StreamWriter(outputFilePath);

            while (sr.EndOfStream == false)
            {
                string? line = sr.ReadLine();
                if (line == null || line.Length == 0)
                {
                    continue;
                }

                dm_r.Clear();
                offsetList.Clear();
                match.Search(line, ref dm_r, ref offsetList, DictMatch.DM_OUT_FMM);

                //if dm_r.Count > 0, it means some contigous terms in strLine have matched terms in the dictionary.
                StringBuilder sb = new StringBuilder();
                int currOffset = 0;

                for (int i = 0; i < dm_r.Count; i++)
                {
                    uint len = dm_r[i].len;
                    int offset = offsetList[i];
                    string strProp = dm_r[i].strProp;
                    string strTerm = line.Substring(offset, (int)len);

                    if (offset > currOffset)
                    {
                        sb.Append(line.Substring(currOffset, offset - currOffset));
                    }

                    sb.Append($" <{strProp}> {strTerm} </{strProp}> ");

                    currOffset = (int)(offset + len);

                }

                if (currOffset < line.Length)
                {
                    sb.Append(line.Substring(currOffset));
                }

                sw.WriteLine(sb.ToString().Replace("  ", " "));
            }
            sr.Close();
            sw.Close();

        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("SeqDictMatchConsole [lexical dictionary file path] [input file path] [output file path]");
                return;
            }


            string dictFilePath = args[0];
            string inputFilePath = args[1];
            string outputFilePath = args[2];

            DictMatchSequences(inputFilePath, outputFilePath, dictFilePath);
        }
    }
}