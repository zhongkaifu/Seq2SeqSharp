using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Seq2SeqWebApps;

namespace SeqWebApps.Pages
{
    public class TextGenerationModel : PageModel
    {
        [BindProperty]
        [Required]
        public string InputText { get; set; }

        [BindProperty]
        public string OutputText { get; set; }

        public void OnGet()
        {
        }

        public IActionResult OnPost()
        {
            string[] lines = InputText.Split(Environment.NewLine);
            List<string> outputLines = new List<string>();

            foreach (var line in lines)
            {
                string inputTextSpm = Utils.SrcSpm.EncodeLine(line.ToLower());
                string outputTextSpm = Seq2SeqInstance.Call(inputTextSpm);
                string outputText = Utils.TgtSpm.DecodeLine(outputTextSpm);

                outputLines.Add(outputText);
            }

            OutputText = String.Join(Environment.NewLine, outputLines);

            return Page();
        }
    }
}
