// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Microsoft.Extensions.Configuration;
using Seq2SeqSharp;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Utils;
using Seq2SeqWebApps;

Logger.LogFile = $"{nameof(SeqWebApps)}_{GetTimeStamp(DateTime.Now)}.log";
Logger.WriteLine($"SeqWebApps v2.7.0 written by Zhongkai Fu(fuzhongkai@gmail.com)");

var Configuration = new ConfigurationBuilder().AddJsonFile("appsettings.json").Build();

if (String.IsNullOrEmpty(Configuration["Seq2Seq:ModelFilePath"]) == false)
{
    Logger.WriteLine($"Loading Seq2Seq model '{Configuration["Seq2Seq:ModelFilePath"]}'");

    var modelFilePath = Configuration["Seq2Seq:ModelFilePath"];
    var maxTestSrcSentLength = String.IsNullOrEmpty(Configuration["Seq2Seq:MaxSrcTokenSize"]) ? 1024 : int.Parse(Configuration["Seq2Seq:MaxSrcTokenSize"]);
    var maxTestTgtSentLength = String.IsNullOrEmpty(Configuration["Seq2Seq:MaxTgtTokenSize"]) ? 1024 : int.Parse(Configuration["Seq2Seq:MaxTgtTokenSize"]);
    var maxTokenToGeneration = String.IsNullOrEmpty(Configuration["Seq2Seq:MaxTokenToGeneration"]) ? 8192 : int.Parse(Configuration["Seq2Seq:MaxTokenToGeneration"]);
    var processorType = String.IsNullOrEmpty(Configuration["Seq2Seq:ProcessorType"]) ? ProcessorTypeEnums.CPU : (Configuration["Seq2Seq:ProcessorType"].ToEnum<ProcessorTypeEnums>());
    var deviceIds = String.IsNullOrEmpty(Configuration["Seq2Seq:DeviceIds"]) ? "0" : Configuration["Seq2Seq:DeviceIds"];
    var decodingStrategyEnum = String.IsNullOrEmpty(Configuration["Seq2Seq:TokenGenerationStrategy"]) ? DecodingStrategyEnums.Sampling : Configuration["Seq2Seq:TokenGenerationStrategy"].ToEnum<DecodingStrategyEnums>();
    var gpuMemoryUsageRatio = String.IsNullOrEmpty(Configuration["Seq2Seq:GPUMemoryUsageRatio"]) ? 0.99f : float.Parse(Configuration["Seq2Seq:GPUMemoryUsageRatio"]);
    var mklInstructions = String.IsNullOrEmpty(Configuration["Seq2Seq:MKLInstructions"]) ? "" : Configuration["Seq2Seq:MKLInstructions"];
    var beamSearchSize = String.IsNullOrEmpty(Configuration["Seq2Seq:BeamSearchSize"]) ? 1 : int.Parse(Configuration["Seq2Seq:BeamSearchSize"]);
    var blockedTokens = String.IsNullOrEmpty(Configuration["Seq2Seq:BlockedTokens"]) ? "" : Configuration["Seq2Seq:BlockedTokens"];
    var modelType = String.IsNullOrEmpty(Configuration["Seq2Seq:ModelType"]) ? ModelType.EncoderDecoder : Configuration["Seq2Seq:ModelType"].ToEnum<ModelType>();
    var wordMappingFilePath = Configuration["Seq2Seq:WordMappingFilePath"];
    var enableTensorCore = string.IsNullOrEmpty(Configuration["Seq2Seq:EnableTensorCore"]) ? true : bool.Parse(Configuration["Seq2Seq:EnableTensorCore"]);
    var compilerOptions = Configuration["Seq2Seq:CompilerOptions"];
    var amp = String.IsNullOrEmpty(Configuration["Seq2Seq:AMP"]) ? false : bool.Parse(Configuration["Seq2Seq:AMP"]);

    Logger.Verbose = String.IsNullOrEmpty(Configuration["Seq2Seq:LogVerbose"]) ? Logger.LogVerbose.Normal : Configuration["Seq2Seq:LogVerbose"].ToEnum<Logger.LogVerbose>();

    SentencePiece? srcSpm = null;
    if (String.IsNullOrEmpty(Configuration["SourceSpm:ModelFilePath"]) == false)
    {
        srcSpm = new SentencePiece(Configuration["SourceSpm:ModelFilePath"]);
    }

    SentencePiece? tgtSpm = null;
    if (String.IsNullOrEmpty(Configuration["TargetSpm:ModelFilePath"]) == false)
    {
        tgtSpm = new SentencePiece(Configuration["TargetSpm:ModelFilePath"]);
    }

    Seq2SeqInstance.Initialization(modelFilePath,
                                   maxTestSrcSentLength,
                                   maxTestTgtSentLength,
                                   maxTokenToGeneration,
                                   processorType,
                                   deviceIds,
                                   srcSpm,
                                   tgtSpm,
                                   decodingStrategyEnum,
                                   memoryUsageRatio: gpuMemoryUsageRatio,
                                   mklInstructions: mklInstructions,
                                   beamSearchSize: beamSearchSize,
                                   blockedTokens: blockedTokens,
                                   modelType: modelType,
                                   wordMappingFilePath: wordMappingFilePath,
                                   enableTensorCore: enableTensorCore,
                                   compilerOptions: compilerOptions,
                                   amp: amp);
}



var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}");

app.Run();


static string GetTimeStamp(DateTime timeStamp)
{
    return string.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
}

/// <summary>
/// 
/// </summary>
internal static class Extensions
{
    public static T ToEnum<T>(this string s) where T : struct => Enum.Parse<T>(s, true);
    public static int ToInt(this string s) => int.Parse(s);
}