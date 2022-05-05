using AdvUtils;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;
using Seq2SeqSharp;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Utils;
using Seq2SeqWebAPI;
using SeqClassificationWebAPI;
using SeqSimilarityWebAPI;
using System;
using System.Collections.Generic;

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

                var srcSentPiece = new SentencePiece( Configuration[ "Seq2Seq:SrcSentencePieceModelPath" ] );
                var tgtSentPiece = new SentencePiece( Configuration[ "Seq2Seq:TgtSentencePieceModelPath" ] );

                Seq2SeqInstance.Initialization( modelFilePath, maxTestSrcSentLength, maxTestTgtSentLength, 
                                                processorType, deviceIds, (srcSentPiece, tgtSentPiece) );
            }

            if ( !Configuration[ "SeqClassification:ModelFilePath" ].IsNullOrEmpty() )
            {
                Logger.WriteLine( $"Loading SeqClassification model '{Configuration[ "SeqClassification:ModelFilePath" ]}'" );

                var modelFilePath = Configuration[ "SeqClassification:ModelFilePath" ];
                int maxTestSentLength = Configuration[ "SeqClassification:MaxTokenSize" ].ToInt();
                processorType = Configuration[ "SeqClassification:ProcessorType" ].ToEnum< ProcessorTypeEnums >();
                deviceIds = Configuration[ "SeqClassification:DeviceIds" ];

                SeqClassificationInstance.Initialization( modelFilePath, maxTestSentLength, processorType, deviceIds );
            }

            if ( !Configuration[ "SeqSimilarity:ModelFilePath" ].IsNullOrEmpty() )
            {
                Logger.WriteLine( $"Loading SeqSimilarity model '{Configuration[ "SeqSimilarity:ModelFilePath" ]}'" );

                var modelFilePath = Configuration[ "SeqSimilarity:ModelFilePath" ];
                int maxTestSentLength = Configuration[ "SeqSimilarity:MaxTokenSize" ].ToInt();
                processorType = Configuration[ "SeqSimilarity:ProcessorType" ].ToEnum< ProcessorTypeEnums >();
                deviceIds = Configuration[ "SeqSimilarity:DeviceIds" ];

                SeqSimilarityInstance.Initialization( modelFilePath, maxTestSentLength, processorType, deviceIds );
            }

            //Loading Seq2SeqClassification models
            if ( !Configuration[ "Seq2SeqClassification:ProcessorType" ].IsNullOrEmpty() )
            {
                int i = 0;
                Dictionary<string, string> key2ModelFilePath = new Dictionary<string, string>();
                while ( true )
                {
                    string key = $"Seq2SeqClassification:Models:{i}:Key";
                    string filePath = $"Seq2SeqClassification:Models:{i}:FilePath";
                    if ( Configuration[ key ].IsNullOrEmpty() )
                    {
                        break;
                    }
                    key2ModelFilePath.Add( Configuration[ key ], Configuration[ filePath ] );

                    i++;
                }

                maxTestSrcSentLength = Configuration[ "Seq2SeqClassification:MaxSrcTokenSize" ].ToInt();
                maxTestTgtSentLength = Configuration[ "Seq2SeqClassification:MaxTgtTokenSize" ].ToInt();
                processorType = Configuration[ "Seq2SeqClassification:ProcessorType" ].ToEnum< ProcessorTypeEnums >();
                deviceIds = Configuration[ "Seq2SeqClassification:DeviceIds" ];

                if ( key2ModelFilePath.Count > 0 )
                {
                    Seq2SeqClassificationInstances.Initialization( key2ModelFilePath, maxTestSrcSentLength, maxTestTgtSentLength, processorType, deviceIds );
                }
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
