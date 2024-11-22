using AdvUtils;
using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;
using System.Diagnostics;

List<string> inputSentences = ["Chest pain with numbness",
"Pressure or tightness in the chest",
"Excessive sweating",
"fever and chills",
"headache and body pain",
"cough and nausia",
"blood in urine",
"fever and chills",
"heart murmur",
"bumps on hands",
"nausia"];
// Cannot contain spaces
List<string> labels = ["heart_attack",
"heart_attack",
"heart_attack",
"flu",
"flu",
"flu",
"endocarditis",
"endocarditis",
"endocarditis",
"endocarditis",
"endocarditis"];

var training = false; // Toggle depending if a retraining is needed (did the statements change?)
do
{
  string prompt = null;
  if (!training)
  {
    Console.WriteLine("What are your symptoms?");
    prompt = Console.ReadLine();
  }
  var rootDir = "D:\\Temp\\";
  // Define model parameters
  var opts = new SeqClassificationOptions
  {
    ModelFilePath = training ? $"{rootDir}model.bin" : $"{rootDir}model.bin.100",
    SrcLang = "Src",
    TgtLang = "Labels",
    ProcessorType = ProcessorTypeEnums.CPU,
    TrainCorpusPath = $"{rootDir}train",
    ValidCorpusPaths = $"{rootDir}valid",
    LogDestination = Logger.Destination.Console,
    TaskParallelism = 4,

    MaxEpochNum = 100,
    //MaxSentLength = 1024,
    //StartLearningRate = 0.01f,
    EncoderLayerDepth = 6, // Increases the model size

    Task = training ? ModeEnums.Train : ModeEnums.Test,
    InputTestFile = $"{rootDir}input.txt",
    OutputFile = $"{rootDir}output.txt",
    LogLevel = training ? Logger.Level.info : Logger.Level.err,
  };
  Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");

  DecodingOptions decodingOptions = opts.CreateDecodingOptions();
  SeqClassification ss = null;

  if (opts.Task == ModeEnums.Train)
  {
    // Save each labels with a tab and then each inputSentences to D:\Temp\train.enu.snt
    Directory.CreateDirectory(opts.TrainCorpusPath);
    File.WriteAllLines($"{opts.TrainCorpusPath}\\train.src.snt", inputSentences.Select((sentence, index) => $"{sentence}"));
    File.WriteAllLines($"{opts.TrainCorpusPath}\\train.labels.snt", inputSentences.Select((sentence, index) => $"{labels[index]}"));

    Directory.CreateDirectory(opts.ValidCorpusPaths);
    File.WriteAllLines($"{opts.ValidCorpusPaths}\\valid.src.snt", inputSentences.Select((sentence, index) => $"{sentence}"));
    File.WriteAllLines($"{opts.ValidCorpusPaths}\\valid.labels.snt", inputSentences.Select((sentence, index) => $"{labels[index]}"));

    // Prepare data for Seq2SeqSharp
    // Load train corpus
    var trainCorpus = new SeqClassificationMultiTasksCorpus(corpusFilePath: opts.TrainCorpusPath,
      srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
      maxSentLength: opts.MaxSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence);

    // Valid corpus
    var validCorpusList = new List<SeqClassificationMultiTasksCorpus>();
    validCorpusList.Add(new SeqClassificationMultiTasksCorpus(opts.ValidCorpusPaths,
      srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, opts.ValMaxTokenSizePerBatch,
      opts.MaxSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence));

    // Create optimizer
    IOptimizer optimizer = Misc.CreateOptimizer(opts);

    var (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize);

    // Create metrics
    Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
    taskId2metrics.Add(0, new List<IMetric>());
    taskId2metrics[0].Add(new MultiLabelsFscoreMetric("", tgtVocab.GetAllTokens(keepBuildInTokens: false)));

    // Create learning rate
    ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount,
      opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

    // Train the model
    ss = new SeqClassification(opts, srcVocab, tgtVocab);

    // Add event handler for monitoring
    ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
    ss.EvaluationWatcher += Ss_EvaluationWatcher;

    ss.Train(opts.MaxEpochNum, trainCorpus, validCorpusList.ToArray(),
    learningRate, taskId2metrics: taskId2metrics, optimizer, decodingOptions);

    Console.WriteLine("Training complete!");
  }
  else if (opts.Task == ModeEnums.Test)
  {
    File.WriteAllText(opts.InputTestFile, prompt);

    if (File.Exists(opts.OutputFile))
    {
      Logger.WriteLine(Logger.Level.info, ConsoleColor.Yellow, $"Output file '{opts.OutputFile}' exist. Delete it.");
      File.Delete(opts.OutputFile);
    }

    //Test trained model
    ss = new SeqClassification(opts);
    Stopwatch stopwatch = Stopwatch.StartNew();

    ss.Test<SeqClassificationMultiTasksCorpusBatch>(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath);

    stopwatch.Stop();

    Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");

    Console.WriteLine(File.ReadAllText(opts.OutputFile));
  }
  void Ss_EvaluationWatcher(object sender, EventArgs e)
  {
    EvaluationEventArg ep = e as EvaluationEventArg;

    Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);
  }
  if (opts.Task == ModeEnums.Train)
  {
    break;
  }
}
while (!training);

