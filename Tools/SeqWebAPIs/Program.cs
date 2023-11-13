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
using TensorSharp;

namespace SeqWebAPIs
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Logger.Initialize(Logger.Destination.Console | Logger.Destination.Logfile, Logger.Level.err | Logger.Level.warn | Logger.Level.info | Logger.Level.debug, $"{nameof(SeqWebAPIs)}_{Utils.GetTimeStamp(DateTime.Now)}.log");

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
