using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

        [HttpGet("{input}")]
        public string Get(string input)
        {
            return Seq2SeqInstance.Call(input);         
        }
    }
}
