using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdvUtils;

namespace Seq2SeqWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class Seq2SeqController : ControllerBase
    {
        private readonly ILogger<Seq2SeqController> _logger;

        public Seq2SeqController(ILogger<Seq2SeqController> logger)
        {
            _logger = logger;
        }


        /// <summary>
        /// Translate API
        /// Try it: curl -X GET "http://localhost:5000/Seq2Seq/Translate?input=%E2%96%81microsoft%20%E2%96%81co%20-%20founder%20%E2%96%81bill%20%E2%96%81gates%20%E2%96%81and%20%E2%96%81his%20%E2%96%81wife%20%E2%96%81of%20%E2%96%8127%20%E2%96%81years%20%E2%96%81melinda%20%E2%96%81gates%20%E2%96%81said%20%E2%96%81in%20%E2%96%81a%20%E2%96%81statement%20%E2%96%81on%20%E2%96%81monday%20%E2%96%81that%20%E2%96%81they%20%E2%96%81intended%20%E2%96%81to%20%E2%96%81end%20%E2%96%81their%20%E2%96%81marriage%20."
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        [HttpGet("Translate")]
        public string Translate(string input)
        {
            string output = Seq2SeqInstance.Call(input);

            Logger.WriteLine($"'{input}' -> '{output}'");

            return output;
        }
    }
}
