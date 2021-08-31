using AdvUtils;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Seq2SeqSharp.Utils;
using SeqClassificationWebAPI;
using SeqSimilarityWebAPI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeqWebAPIs
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Logger.LogFile = $"{nameof(SeqWebAPIs)}_{Utils.GetTimeStamp(DateTime.Now)}.log";

            if (args.Length != 6)
            {
                Logger.WriteLine(Logger.Level.err, $"SeqWebAPIs.exe [tsc model file path] [sentSim model file path] [hpi model file path] [exam model file path] [assessment model file path] [results model file path]");
                return;
            }

            string tscModelFilePath = args[0];
            string sentSimModelFilePath = args[1];
            string hpiModelFilePath = args[2];
            string examModelFilePath = args[3];
            string assessmentModelFilePath = args[4];
            string resModelFilePath = args[5];


            SeqClassificationInstance.Initialization(tscModelFilePath, 512, "CPU", "0");
            SeqSimilarityInstance.Initialization(sentSimModelFilePath, 512, "CPU", "0");

            Dictionary<string, string> key2ModelFilePath = new Dictionary<string, string>();
            key2ModelFilePath.Add("hpi", hpiModelFilePath);
            key2ModelFilePath.Add("pe", examModelFilePath);
            key2ModelFilePath.Add("ap", assessmentModelFilePath);
            key2ModelFilePath.Add("res", resModelFilePath);

            Seq2SeqClassificationInstances.Initialization(key2ModelFilePath, 1024, 256, "CPU", "0");


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
