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

        [HttpGet("Classify")]
        public string Classify(string inFeature1, string inFeature2)
        {
            var output = SeqClassificationInstance.Call(inFeature1, inFeature2);

            Logger.WriteLine($"'{inFeature1}' | '{inFeature2}' -> '{output}'");
            return output;
        }
    }
}
