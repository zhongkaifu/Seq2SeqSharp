// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Text;
using AdvUtils;
using Microsoft.AspNetCore.Mvc;
using Seq2SeqWebApps;
using SeqWebApps.Models;

namespace SeqWebApps.Controllers
{
    public class HomeController : Controller
    {
        static Dictionary<string, string> dictInputSents = new Dictionary<string, string>();
        private static DateTime m_dtLastDumpLogs = DateTime.Now;
        private static object locker = new object();

        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult GenerateText(string srcInput, string tgtInput, int num, bool random, float repeatPenalty, int contextSize, string clientIP, bool useSrcAsPrompt)
        {
            if (tgtInput == null)
            {
                tgtInput = "";
            }

            if (useSrcAsPrompt && String.IsNullOrEmpty(tgtInput))
            {
                Logger.WriteLine($"Using source text '{srcInput}' as prompt to target text.");
                tgtInput = srcInput;
            }

            TextGenerationModel textGeneration = new TextGenerationModel
            {
                Output = CallBackend(srcInput, tgtInput, num, random, repeatPenalty, contextSize, clientIP),
                DateTime = DateTime.Now.ToString()
            };

            return new JsonResult(textGeneration);
        }


        private string CallBackend(string srcInputText, string tgtInputText, int tokenNumToGenerate, bool random, float repeatPenalty, int tgtContextSize, string clientIP)
        {
            if (String.IsNullOrEmpty(srcInputText))
            {
                srcInputText = "";
            }

            if (String.IsNullOrEmpty(tgtInputText))
            {
                tgtInputText = "";
            }


            srcInputText = srcInputText.Replace("<br />", "");
            tgtInputText = tgtInputText.Replace("<br />", "");

            string[] srcLines = srcInputText.Split("\n");
            string[] tgtLines = tgtInputText.Split("\n");

            srcInputText = String.Join("", srcLines);
            tgtInputText = String.Join("", tgtLines);


            string prefixTgtLine = "";

            //The generated target tokens are too long, let's truncate it.
            var maxTgtContextSize = tgtContextSize - tokenNumToGenerate;
            if (tgtInputText.Length > maxTgtContextSize)
            {
                int idx = tgtInputText.Length - maxTgtContextSize;
                while (idx > 0)
                {
                    if (tgtInputText[idx] == '。' || tgtInputText[idx] == '.' || tgtInputText[idx] == '？' || tgtInputText[idx] == '!' || tgtInputText[idx] == '?' || tgtInputText[idx] == '!')
                    {
                        idx++;
                        break;
                    }
                    idx--;
                }

                prefixTgtLine = tgtInputText.Substring(0, idx);
                tgtInputText = tgtInputText.Substring(idx);
            }

            string logStr = $"Client = '{clientIP}', SrcInput Text = '{srcInputText}', Repeat Penalty = '{repeatPenalty}', Target Context Size = '{tgtContextSize}'";
            lock (locker)
            {
                if (dictInputSents.ContainsKey(logStr) == false)
                {
                    Logger.WriteLine(logStr);
                }

            }

            string outputText = Seq2SeqInstance.Call(srcInputText, tgtInputText, tokenNumToGenerate, random, repeatPenalty);
            outputText = prefixTgtLine + outputText;

            // Update logs and dump it every 1 hour when a call comes in.
            lock (locker)
            {
                string truncatedOutput = outputText.Replace(srcInputText, "");
                if (dictInputSents.ContainsKey(logStr) == false)
                {
                    dictInputSents.Add(logStr, truncatedOutput);
                }
                else
                {
                    dictInputSents[logStr] = truncatedOutput;
                }
                if (DateTime.Now - m_dtLastDumpLogs >= TimeSpan.FromHours(1.0))
                {
                    string dumpFilePath = Path.Combine(Directory.GetCurrentDirectory(), "dump_generated_text.log");
                    List<string> dumpList = new List<string>();
                    foreach (var pair in dictInputSents)
                    {
                        Logger.WriteLine($"Key = '{pair.Key}', Value = '{pair.Value}'");

                        dumpList.Add($"Source = '{pair.Key}'");
                        dumpList.Add($"Generated text = '{pair.Value}'");
                        dumpList.Add("");
                    }

                    System.IO.File.AppendAllLines(dumpFilePath, dumpList);

                    dictInputSents.Clear();

                    m_dtLastDumpLogs = DateTime.Now;
                }
            }


            var outputSents = SplitSents(outputText);
            return String.Join("<br />", outputSents);

        }

        private static string[] Split(string text, char[] seps)
        {
            HashSet<char> setSeps = new HashSet<char>();
            foreach (var sep in seps)
            {
                setSeps.Add(sep);
            }

            List<string> parts = new List<string>();
            StringBuilder sb = new StringBuilder();
            foreach (char ch in text)
            {
                sb.Append(ch);
                if (setSeps.Contains(ch) && sb.Length > 1)
                {
                    parts.Add(sb.ToString().Trim());
                    sb = new StringBuilder();
                }
            }

            if (sb.Length > 0)
            {
                parts.Add(sb.ToString());
            }

            return parts.ToArray();
        }

        private List<string> SplitSents(string currentSent)
        {
            List<string> sents = new List<string>();

            HashSet<char> setClosedPunct = new HashSet<char>();
            setClosedPunct.Add('”');
            setClosedPunct.Add('\"');
            setClosedPunct.Add('】');
            setClosedPunct.Add(')');
        
            string[] parts = Split(currentSent, new char[] { '。', '！', '?', '!', '?' });
            for (int i = 0; i < parts.Length; i++)
            {
                string p = String.Empty;
                bool skipNextLine = false;
                if (i < parts.Length - 1)
                {
                    if (setClosedPunct.Contains(parts[i + 1][0]))
                    {
                        p = parts[i] + parts[i + 1];
                        skipNextLine = true;
                    }
                    else
                    {
                        p = parts[i];
                    }
                }
                else
                {
                    p = parts[i];
                }

                sents.Add(p);

                if (skipNextLine)
                {
                    i++;
                }
            }

            return sents;

            //List<string> newSents = new List<string>();
            //int matchNum = 0;
            //string currSent = "";
            //for (int k = 0; k < sents.Count; k++)
            //{
            //    var sent = sents[k];
            //    for (int i = 0; i < sent.Length; i++)
            //    {
            //        if (sent[i] == '“')
            //        {
            //            matchNum++;
            //        }
            //        else if (sent[i] == '”')
            //        {
            //            matchNum--;
            //        }
            //    }

            //    currSent = currSent + sent;
            //    if (matchNum == 0)
            //    {
            //        newSents.Add(currSent);
            //        currSent = "";
            //    }
            //}

            //newSents.Add(currSent);

            //return newSents;
        }

    }
}
