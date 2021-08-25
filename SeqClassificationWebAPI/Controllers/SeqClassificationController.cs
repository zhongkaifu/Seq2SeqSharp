using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
        public string Get(string input)
        {
            return SeqClassificationInstance.Call(input);
        }
    }
}
