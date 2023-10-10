using AdvUtils;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Utils;
using Seq2SeqWebAPI;
using System;

namespace SeqWebAPIs
{
    public class Startup
    {
        public Startup( IConfiguration configuration )
        {
            int maxTestSrcSentLength;
            int maxTestTgtSentLength;
            ProcessorTypeEnums processorType;
            string deviceIds;

            Configuration = configuration;

            if ( !Configuration[ "Seq2Seq:ModelFilePath" ].IsNullOrEmpty())
            {
                Logger.WriteLine( $"Loading Seq2Seq model '{Configuration[ "Seq2Seq:ModelFilePath" ]}'" );

                var modelFilePath = Configuration[ "Seq2Seq:ModelFilePath" ];
                maxTestSrcSentLength = Configuration[ "Seq2Seq:MaxSrcTokenSize" ].ToInt();
                maxTestTgtSentLength = Configuration[ "Seq2Seq:MaxTgtTokenSize" ].ToInt();
                processorType = Configuration[ "Seq2Seq:ProcessorType" ].ToEnum< ProcessorTypeEnums >();
                deviceIds = Configuration[ "Seq2Seq:DeviceIds" ];

                SentencePiece? srcSpm = null;
                if (String.IsNullOrEmpty(Configuration["Seq2Seq:SrcSentencePieceModelPath"]) == false)
                {
                    srcSpm = new SentencePiece(Configuration["Seq2Seq:SrcSentencePieceModelPath"]);
                }

                SentencePiece? tgtSpm = null;
                if (String.IsNullOrEmpty(Configuration["Seq2Seq:TgtSentencePieceModelPath"]) == false)
                {
                    tgtSpm = new SentencePiece(Configuration["Seq2Seq:TgtSentencePieceModelPath"]);
                }

                Seq2SeqInstance.Initialization( modelFilePath, maxTestSrcSentLength, maxTestTgtSentLength, 
                                                processorType, deviceIds, (srcSpm, tgtSpm) );
            }
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices( IServiceCollection services )
        {

            services.AddControllers();
            services.AddSwaggerGen( c =>
             {
                 c.SwaggerDoc( "v1", new OpenApiInfo { Title = "SeqWebAPIs", Version = "v1" } );
             } );
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure( IApplicationBuilder app, IWebHostEnvironment env )
        {
            if ( env.IsDevelopment() )
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI( c => c.SwaggerEndpoint( "/swagger/v1/swagger.json", "SeqWebAPIs v1" ) );
            }

            app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints( endpoints =>
             {
                 endpoints.MapControllers();
             } );
        }
    }

    /// <summary>
    /// 
    /// </summary>
    internal static class Extensions
    {
        public static T ToEnum< T >( this string s ) where T : struct => Enum.Parse< T >( s, true );
        public static int ToInt( this string s ) => int.Parse( s );
    }
}
