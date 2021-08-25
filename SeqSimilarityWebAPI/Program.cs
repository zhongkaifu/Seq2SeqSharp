using AdvUtils;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeqSimilarityWebAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Logger.LogFile = $"{nameof(SeqSimilarityWebAPI)}_{Utils.GetTimeStamp(DateTime.Now)}.log";

            var modelFilePath = args[0];
            var maxTestSentLength = int.Parse(args[1]);
            var processorType = args[2];

            SeqSimilarityInstance.Initialization(modelFilePath, maxTestSentLength, processorType);

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
