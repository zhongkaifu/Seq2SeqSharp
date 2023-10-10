﻿using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Layers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Seq2SeqSharp.Applications
{
    public static class ImgEncoder
    {
        static int IMAGE_W = 256;
        static int IMAGE_H = 256;

        static int TOKEN_W = 16;
        static int TOKEN_H = 16;

        static int TOTAL_TOKEN_NUM_PER_IMG = (IMAGE_W / TOKEN_W) * (IMAGE_H / TOKEN_H);
    
        static private IWeightTensor LoadImageToTokens(IComputeGraph g, string filePath, bool rotation = false, bool horizFlip = false, bool vertFlip = false, bool brigtness = false, bool contrast = false)
        {

            List<float[]> tokens = new List<float[]>();

            using (Image<Rgb24> image = Image.Load<Rgb24>(filePath))
            {
                if (rotation == true)
                {
                    image.Mutate(x => x.Rotate(90.0f));
                }

                if (horizFlip == true)
                {
                    image.Mutate(x => x.Flip(FlipMode.Horizontal));
                }

                if (vertFlip == true)
                {
                    image.Mutate(x => x.Flip(FlipMode.Vertical));
                }

                if (brigtness == true)
                {
                    image.Mutate(x => x.Brightness(0.5f));
                }

                if (contrast == true)
                {
                    image.Mutate(x => x.Contrast(0.5f));
                }

                int newWidth = 0;
                int newHeight = 0;
                if (image.Width < image.Height)
                {
                    newWidth = IMAGE_W;
                    newHeight = (newWidth / image.Width) * image.Height;
                }
                else
                {
                    newHeight = IMAGE_H;
                    newWidth = (newHeight / image.Height) * image.Width;
                }

                image.Mutate(x => x.Resize(newWidth, newHeight));

                image.Mutate(x =>
                {
                    x.Resize(new ResizeOptions
                    {
                        Size = new Size(IMAGE_W, IMAGE_W),
                        Mode = ResizeMode.Crop
                    });
                });


                var mean = new[] { 0.485f, 0.456f, 0.406f };
                var stddev = new[] { 0.229f, 0.224f, 0.225f };
                float[] processedImage = new float[IMAGE_W * IMAGE_H * 3];

                image.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (int x = 0; x < accessor.Width; x++)
                        {
                            int offset = (y * accessor.Width + x) * 3;

                            processedImage[offset] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                            processedImage[offset + 1] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                            processedImage[offset + 2] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                        }
                    }
                });


                IWeightTensor res = g.CreateTensorWeights(sizes: new long[] {IMAGE_W / TOKEN_W, TOKEN_W, IMAGE_H / TOKEN_H, TOKEN_H, 3 }, processedImage);
                res = g.AsContiguous(g.Transpose(res, 1, 2)); // shape: [IMAGE_W / TOKEN_W, IMAGE_H / TOKEN_H, TOKEN_W, TOKEN_H, 3]
                res = g.View(res, dims: new long[] { -1, 768 });

                return res;                
            } 
        }



        //Tensor Shape: [batchsize, token, embedding_dim]
        //Size(token) = TOTAL_TOKEN_NUM_PER_IMG
        //Size(embedding_dim) = 768
        //Shape: [batchsize, TOTAL_TOKEN_NUM_PER_IMG, 768]
        static private IWeightTensor InnerEncode(IComputeGraph g, List<string> imgPaths, bool trainMode = false)
        {
            int batchSize = imgPaths.Count;
            List<IWeightTensor> batchTokens = new List<IWeightTensor>();

            foreach (var picPath in imgPaths)
            {
                batchTokens.Add(LoadImageToTokens(g, picPath)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  

                if (trainMode)
                {
                    batchTokens.Add(LoadImageToTokens(g, picPath, rotation: true)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  
                    batchTokens.Add(LoadImageToTokens(g, picPath, horizFlip: true)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  
                    batchTokens.Add(LoadImageToTokens(g, picPath, vertFlip: true)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  
                    batchTokens.Add(LoadImageToTokens(g, picPath, brigtness: true)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  
                    batchTokens.Add(LoadImageToTokens(g, picPath, contrast: true)); //shape: [TOTAL_TOKEN_NUM_PER_IMG, 768]  
                }
            }

            var res = g.Concate(batchTokens, 0);
            return res;            
        }

        static public IWeightTensor Run(IComputeGraph g, List<string> imgPaths, IEncoder encoder, IFeedForwardLayer srcEmbeddings, IWeightTensor posEmbeddings, INormalization layerNorm1, INormalization layerNorm2, IWeightTensor cls, bool trainMode = false)
        {
            int batchSize = imgPaths.Count;
            var inputEmbs = InnerEncode(g, imgPaths, trainMode);

            if (trainMode)
            {
                batchSize = batchSize * 6;
            }


        //    inputEmbs = layerNorm1.Norm(inputEmbs, g);
            inputEmbs = srcEmbeddings.Process(inputEmbs, batchSize, g);
            //     inputEmbs = g.ReLU  (inputEmbs);
            //   inputEmbs = layerNorm2.Norm(inputEmbs, g);

            inputEmbs = g.View(inputEmbs, dims: new long[] { batchSize, -1, 768 });

            cls = g.View(cls, dims: new long[] { 1, 1, 768 });
            cls = g.Expand(cls, dims: new long[] { batchSize, 1, 768 });

            inputEmbs = g.Concate(1, cls, inputEmbs);

            inputEmbs = g.View(inputEmbs, dims: new long[] { -1, 768 });



            inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbeddings, batchSize, inputEmbs, 0.0f);           
            return encoder.Encode(inputEmbs, batchSize, g, null);
        }
    }
}