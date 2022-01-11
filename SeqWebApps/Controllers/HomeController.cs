using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AdvUtils;
//using System.Web.Mvc;
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
        public IActionResult GenerateText(string srcInput, string tgtInput, int num, bool random, float distancePenalty, float repeatPenalty, int contextSize)
        {
            TextGenerationModel textGeneration = new TextGenerationModel
            {
                Output = CallBackend(srcInput, tgtInput, num, random, distancePenalty, repeatPenalty, contextSize),
                DateTime = DateTime.Now.ToString()
            };

            return new JsonResult(textGeneration);
        }


        private string CallBackend(string srcInputText, string tgtInputText, int tokenNumToGenerate, bool random, float distancePenalty, float repeatPenalty, int tgtContextSize)
        {
            srcInputText = srcInputText.Replace("<br />", "");
            tgtInputText = tgtInputText.Replace("<br />", "");

            string[] srcLines = srcInputText.Split("\n");
            string[] tgtLines = tgtInputText.Split("\n");

            srcInputText = String.Join(" ", srcLines).ToLower();
            tgtInputText = String.Join(" ", tgtLines).ToLower();


            string prefixTgtLine = "";
            if (tgtInputText.Length > tgtContextSize)
            {
                prefixTgtLine = tgtInputText.Substring(0, tgtInputText.Length - tgtContextSize);
                tgtInputText = tgtInputText.Substring(tgtInputText.Length - tgtContextSize);
            }

            Stopwatch stopwatch = Stopwatch.StartNew();

            if (srcInputText.EndsWith("。") == false && srcInputText.EndsWith("？") == false && srcInputText.EndsWith("！") == false)
            {
                srcInputText = srcInputText + "。";
            }

            if (setInputSents.Contains(srcInputText) == false)
            {
                Logger.WriteLine(srcInputText);
                setInputSents.Add(srcInputText);
            }

            string outputText = Seq2SeqInstance.Call(srcInputText, tgtInputText, tokenNumToGenerate, random, distancePenalty, repeatPenalty);

            stopwatch.Stop();

            outputText = prefixTgtLine + outputText;

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
                    parts.Add(sb.ToString());
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

            string[] parts = Split(currentSent, new char[] { '。', '！', '?' });
            for (int i = 0; i < parts.Length; i++)
            {
                string p = String.Empty;
                if (i < parts.Length - 1)
                {
                    if (parts[i + 1][0] == '”')
                    {
                        parts[i + 1] = parts[i + 1].Substring(1);
                        p = parts[i] + "”";
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
            }

            return sents;
        }

    }
}
