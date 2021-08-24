using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Seq2SeqWebAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var modelFilePath = args[0];
            var maxTestSrcSentLength = int.Parse(args[1]);
            var maxTestTgtSentLength = int.Parse(args[2]);
            var processorType = args[3];

            Seq2SeqInstance.Initialization(modelFilePath, maxTestSrcSentLength, maxTestTgtSentLength, processorType);

            CreateHostBuilder(args).Build().Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
