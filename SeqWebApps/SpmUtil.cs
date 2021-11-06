using System.Diagnostics;
using System.Text;

namespace Seq2SeqWebApps
{
    public class SpmUtils
    {
        Process pEncode;
        Process pDecode;

        public SpmUtils(string modelFilePath)
        {
            ProcessStartInfo psiEncode = new ProcessStartInfo();
            psiEncode.FileName = @"spm_encode.exe";
            psiEncode.Arguments = $"-model {modelFilePath}";
            psiEncode.UseShellExecute = false;

            psiEncode.RedirectStandardOutput = true;
            psiEncode.RedirectStandardInput = true;
            psiEncode.RedirectStandardError = true;


            psiEncode.StandardOutputEncoding = Encoding.UTF8;
            psiEncode.StandardErrorEncoding = Encoding.UTF8;


            pEncode = new Process();
            pEncode.StartInfo = psiEncode;
            pEncode.Start();



            ProcessStartInfo psiDecode = new ProcessStartInfo();
            psiDecode.FileName = @"spm_decode.exe";
            psiDecode.Arguments = $"-model {modelFilePath}";
            psiDecode.UseShellExecute = false;

            psiDecode.RedirectStandardOutput = true;
            psiDecode.RedirectStandardInput = true;
            psiDecode.RedirectStandardError = true;

            psiDecode.StandardOutputEncoding = Encoding.UTF8;
            psiDecode.StandardErrorEncoding = Encoding.UTF8;

            pDecode = new Process();
            pDecode.StartInfo = psiDecode;
            pDecode.Start();

        }

        public string EncodeLine(string input)
        {
            pEncode.StandardInput.WriteLine(input);
            pEncode.StandardInput.Flush();

            return pEncode.StandardOutput.ReadLine();
        }

        public string DecodeLine(string input)
        {
            pDecode.StandardInput.WriteLine(input);
            pDecode.StandardInput.Flush();

            return pDecode.StandardOutput.ReadLine();
        }
    }
}