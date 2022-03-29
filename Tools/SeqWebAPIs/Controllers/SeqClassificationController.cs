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

        [HttpGet("{input}")]
        public string Classify(string input)
        {
            List<string> inputGroups = input.Split('\t').ToList();
            var output = SeqClassificationInstance.Call(inputGroups);

            Logger.WriteLine($"'{input}' -> '{output}'");
            return output;
        }
    }
}
