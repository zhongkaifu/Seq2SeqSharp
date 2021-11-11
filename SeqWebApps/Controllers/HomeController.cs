using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdvUtils;
//using System.Web.Mvc;
using Microsoft.AspNetCore.Mvc;
using Seq2SeqWebApps;

namespace SeqWebApps
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public JsonResult GenerateText(string input)
        {
            TextGenerationModel textGeneration = new TextGenerationModel
            {
                Output = CallBackend(input),
                DateTime = DateTime.Now.ToString()
            };

            return Json(textGeneration);
        }


        private string CallBackend(string InputText)
        {
            string[] lines = InputText.Split(Environment.NewLine);
            List<string> outputLines = new List<string>();

            foreach (var line in lines)
            {
                string inputTextSpm = Utils.SrcSpm.EncodeLine(line.ToLower());
                string outputTextSpm = Seq2SeqInstance.Call(inputTextSpm);
                string outputText = Utils.TgtSpm.DecodeLine(outputTextSpm);

                outputLines.Add(outputText);

                Logger.WriteLine($"{line} --> {outputText}");
            }

            return String.Join("<br />", outputLines);
        }

    }
}
