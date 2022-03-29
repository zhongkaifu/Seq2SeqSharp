using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdvUtils;
using SeqWebAPIs;

namespace Seq2SeqClassificationWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class Seq2SeqClassificationController : ControllerBase
    {

        private readonly ILogger<Seq2SeqClassificationController> _logger;

        public Seq2SeqClassificationController(ILogger<Seq2SeqClassificationController> logger)
        {
            _logger = logger;
        }

        [HttpGet("{key}/{input}")]
        public string Get(string key, string input)
        {
            List<string> inputGroups = input.Split('\t').ToList();
            (var tag, var text) = Seq2SeqClassificationInstances.Call(key, inputGroups);

            Logger.WriteLine($"'{key}' | '{input}' -> '{tag}' | '{text}'");
            return tag + "\t" + text;
        }
    }
}
