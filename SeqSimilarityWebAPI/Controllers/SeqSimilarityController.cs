using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdvUtils;

namespace SeqSimilarityWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SeqSimilarityController : ControllerBase
    {
        private readonly ILogger<SeqSimilarityController> _logger;

        public SeqSimilarityController(ILogger<SeqSimilarityController> logger)
        {
            _logger = logger;
        }

        [HttpGet("{input1}/{input2}")]
        public string Similarity(string input1, string input2)
        {
            var output = SeqSimilarityInstance.Call(input1, input2);

            Logger.WriteLine($"'{input1}' | '{input2}' -> '{output}'");
            return output;
        }
    }
}
