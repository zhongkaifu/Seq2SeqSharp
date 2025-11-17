# Vision Caption Console

The **VisionCaptionConsole** project is a .NET 8 command-line app that exposes Seq2SeqSharp's vision encoder + text decoder pipeline. It lets you train, validate, and test transformer-based image captioning models directly from aligned image/caption corpora without touching the legacy `Image2Seq` stack.

## Prerequisites

- [.NET SDK 8.0+](https://dotnet.microsoft.com/download) on your PATH.
- A CUDA-capable GPU and the native dependencies described in the root `README.md` if you plan to train large models. CPU-only runs are supported for experimentation but are significantly slower.
- Training/validation corpora where **each source line contains an image path** and the matching **target line contains a space-delimited caption**. Multiple corpus files can be specified by separating the paths with semicolons (see the `--TrainCorpusPath` and `--ValidCorpusPaths` options).

## Building

```bash
# From the repository root
 dotnet build Seq2SeqSharp.sln -c Release
```

The solution build restores all required NuGet packages (including `SixLabors.ImageSharp` for image loading) and drops the console binary into `Tools/VisionCaptionConsole/bin/`.

## Quick Start

The console accepts the same option set as the text-based Seq2Seq console, plus a few vision-specific flags defined on `Seq2SeqOptions`:

| Option | Description |
| --- | --- |
| `--VisionImageSize` | Square size, in pixels, that images are resized to before patching (default: `224`). |
| `--VisionPatchSize` | Pixel width/height of each ViT-style patch extracted from the resized image (default: `16`). |
| `--VisionChannelMean` / `--VisionChannelStd` | Comma-separated RGB mean/std vectors used for per-channel normalization. |

> **Tip:** Set `--EncoderType CNN --CnnKernelSize 5` to swap the ViT-style transformer encoder for the new lightweight CNN encoder. Combine it with `--CnnChannelSchedule 768,1024,1280,1024,768` (matching your encoder depth) to widen intermediate layers while keeping the first/last layers aligned with the model hidden size.

### Training

```bash
 dotnet run --project Tools/VisionCaptionConsole/VisionCaptionConsole.csproj -- \
   --Task Train \
   --TrainCorpusPath data/train.src;data/train.tgt \
   --ValidCorpusPaths data/val.src;data/val.tgt \
   --ModelFilePath models/vision_caption.mdl \
   --TgtVocabSize 30000 \
   --BatchSize 64 \
   --MaxTokenSizePerBatch 8192 \
   --DecoderType Transformer \
   --VisionImageSize 224 \
   --VisionPatchSize 16
```

Key behaviors:

- Vocabulary: If `--TgtVocab` is omitted the console builds one from the training corpus and stores it next to `ModelFilePath`.
- Validation: Any `--ValidCorpusPaths` are evaluated at the end of each epoch via BLEU (or ROUGE-L) and Length Ratio metrics.
- Checkpoints: The latest checkpoint is always written to `ModelFilePath`. You can resume training by pointing `--ModelFilePath` to an existing model.

### Validation Only

```bash
 dotnet run --project Tools/VisionCaptionConsole/VisionCaptionConsole.csproj -- \
   --Task Valid \
   --ModelFilePath models/vision_caption.mdl \
   --ValidCorpusPaths data/val.src;data/val.tgt
```

### Testing / Caption Generation

```bash
 dotnet run --project Tools/VisionCaptionConsole/VisionCaptionConsole.csproj -- \
   --Task Test \
   --ModelFilePath models/vision_caption.mdl \
   --InputTestFile data/test.src \
   --OutputFile results/test.captions \
   --BeamSearchSize 5
```

Each line in `InputTestFile` must be a resolvable image path. The tool writes the generated caption per line in `OutputFile`.

## Configuration Files

All CLI flags can be stored in a JSON config and re-used via `--ConfigFilePath`. Below is a minimal example:

```json
{
  "Task": "Train",
  "TrainCorpusPath": "data/train.src;data/train.tgt",
  "ValidCorpusPaths": "data/val.src;data/val.tgt",
  "ModelFilePath": "models/vision_caption.mdl",
  "BatchSize": 64,
  "MaxTokenSizePerBatch": 8192,
  "MaxSrcSentLength": 256,
  "MaxTgtSentLength": 64,
  "VisionImageSize": 224,
  "VisionPatchSize": 16,
  "VisionChannelMean": "0.485,0.456,0.406",
  "VisionChannelStd": "0.229,0.224,0.225"
}
```

Launch the console with:

```bash
 dotnet run --project Tools/VisionCaptionConsole/VisionCaptionConsole.csproj -- --ConfigFilePath config/vision_train.json
```

## Data Preparation Tips

1. Keep all image paths absolute or relative to the working directory where you launch the console.
2. Pre-tokenize captions into space-delimited tokens and lowercase them to stabilize the vocabulary.
3. The encoder follows the Vision Transformer (ViT) patching scheme: each `VisionPatchSize × VisionPatchSize` tile is flattened (RGB order), projected with a learnable linear layer, normalized, and scaled to match the decoder's embedding range. A learnable `[CLS]` token is prepended to provide a global summary of the image. The number of visual tokens is therefore `(VisionImageSize / VisionPatchSize)^2 + 1` (extra 1 for `[CLS]`), so adjust `VisionPatchSize` if your GPU memory is limited (larger patches → fewer tokens per image, smaller patches → richer spatial detail).
4. To reuse a prebuilt vocabulary, set `--TgtVocab` to the `.vocab` file exported from a prior run.

## Support Modes

`VisionCaptionConsole` exposes all Seq2SeqSharp modes relevant to image captioning:

- `Train`: joint training with optional validation.
- `Valid`: run metrics-only evaluation on a corpus.
- `Test`: batch caption generation with greedy/beam/TopP decoding.
- `DumpVocab`: export the vocab embedded in a checkpoint.
- `UpdateVocab`: swap in a different target vocabulary file.
- `VQModel`: run vector quantization on an existing checkpoint.

(Alignment mode is intentionally disabled because image tokens have no 1:1 text span alignment.)

## Logging & Notifications

Logs are stored in `Logs/` (or the directory specified by `--LogDestination`). You can set `--NotifyEmail you@example.com` to forward evaluation summaries; SMTP settings are inherited from the shared Seq2SeqSharp infrastructure.

---
Questions or issues? File a GitHub ticket with your command line, console log, and (if possible) a minimal corpus sample so others can reproduce the problem quickly.
