using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Seq2SeqWebApps;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeqWebApps
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;

            if (String.IsNullOrEmpty(Configuration["Seq2Seq:ModelFilePath"]) == false)
            {
               // Logger.WriteLine($"Loading Seq2Seq model '{Configuration["Seq2Seq:ModelFilePath"]}'");

                var modelFilePath = Configuration["Seq2Seq:ModelFilePath"];
                var maxTestSrcSentLength = int.Parse(Configuration["Seq2Seq:MaxSrcTokenSize"]);
                var maxTestTgtSentLength = int.Parse(Configuration["Seq2Seq:MaxTgtTokenSize"]);
                var processorType = Configuration["Seq2Seq:ProcessorType"];
                var deviceIds = Configuration["Seq2Seq:DeviceIds"];
                var tokenGenerationStrategy = Configuration["Seq2Seq:TokenGenerationStrategy"];

                Seq2SeqInstance.Initialization(modelFilePath, maxTestSrcSentLength, maxTestTgtSentLength, processorType, deviceIds, tokenGenerationStrategy);
            }


            Utils.SrcSpm = new SpmUtils(Configuration["SourceSpm:ModelFilePath"]);
            Utils.TgtSpm = new SpmUtils(Configuration["TargetSpm:ModelFilePath"]);
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddRazorPages();
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
            }

            app.UseStaticFiles();

            app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapRazorPages();
            });
        }
    }
}
