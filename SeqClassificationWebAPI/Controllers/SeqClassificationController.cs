using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdvUtils;

namespace SeqClassificationWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SeqClassificationController : ControllerBase
    {

        private readonly ILogger<SeqClassificationController> _logger;

        public SeqClassificationController(ILogger<SeqClassificationController> logger)
        {
            _logger = logger;
        }

        [HttpGet("{input1}/{input2}")]
        public string Classify(string input1, string input2)
        {
            var output = SeqClassificationInstance.Call(input1, input2);

            Logger.WriteLine($"'{input1}' | '{input2}' -> '{output}'");
            return output;
        }
    }
}
