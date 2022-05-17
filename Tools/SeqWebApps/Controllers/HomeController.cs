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
        static HashSet<string> setInputSents = new HashSet<string>();

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
        public IActionResult GenerateText(string srcInput, string tgtInput, int num, bool random, float repeatPenalty, int contextSize, string clientIP)
        {
            if (tgtInput == null)
            {
                tgtInput = "";
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
            srcInputText = srcInputText.Replace("<br />", "");
            tgtInputText = tgtInputText.Replace("<br />", "");

            string[] srcLines = srcInputText.Split("\n");
            string[] tgtLines = tgtInputText.Split("\n");

            srcInputText = String.Join("", srcLines).ToLower();
            tgtInputText = String.Join("", tgtLines).ToLower();


            string prefixTgtLine = "";

            //The generated target tokens are too long, let's truncate it.
            if (tgtInputText.Length > tgtContextSize)
            {
                prefixTgtLine = tgtInputText.Substring(0, tgtInputText.Length - tgtContextSize);
                tgtInputText = tgtInputText.Substring(tgtInputText.Length - tgtContextSize);

            }

            Stopwatch stopwatch = Stopwatch.StartNew();

            string logStr = $"Client = '{clientIP}', SrcInput Text = '{srcInputText}', Repeat Penalty = '{repeatPenalty}', Target Context Size = '{tgtContextSize}'";
            if (setInputSents.Contains(logStr) == false)
            {
                Logger.WriteLine(logStr);
                setInputSents.Add(logStr);
            }

            string outputText = Seq2SeqInstance.Call(tgtInputText, tgtInputText, tokenNumToGenerate, random, repeatPenalty);

            stopwatch.Stop();

            outputText = prefixTgtLine.Trim() + outputText.Trim();

            outputText = outputText.Trim();

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
