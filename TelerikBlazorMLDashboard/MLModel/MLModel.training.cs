//https://learn.microsoft.com/en-us/azure/open-datasets/dataset-taxi-yellow?tabs=azureml-opendatasets

using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;

namespace BlazorMLDashboard;
public partial class MLModel
{

    /// <summary>
    /// Train a new model with the provided dataset.
    /// </summary>
    public void Train()
    {
        MLContext mlContext = new();
        
        // Load Data
        IDataView data = LoadIDataViewFromFile(mlContext, Settings.RetrainFileName, Settings);
        
        // Transform Data
        // Train Model
        ITransformer model = RetrainModel(mlContext, data);
        
        //Save Model
        using var fs = File.Create(Settings.GetPath(Settings.ModelFileName));
        mlContext.Model.Save(model, data.Schema, fs);
    }

    /// <summary>
    /// Retrain model using the pipeline generated as part of the training process.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="trainData"></param>
    /// <returns></returns>
    private ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
    {
        var pipeline = BuildPipeline(mlContext);
        var model = pipeline.Fit(trainData);

        return model;
    }

    /// <summary>
    /// build the pipeline that is used from model builder. Use this function to retrain model.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <returns></returns>
    private IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        // Data process configuration with pipeline data transformations
        => mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"vendor_id", @"vendor_id"), new InputOutputColumnPair(@"payment_type", @"payment_type") }, outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
                                .Append(mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"rate_code", @"rate_code"), new InputOutputColumnPair(@"passenger_count", @"passenger_count"), new InputOutputColumnPair(@"trip_distance", @"trip_distance") }))
                                .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"vendor_id", @"payment_type", @"rate_code", @"passenger_count", @"trip_distance" }))
                                .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() { NumberOfLeaves = 2456, NumberOfIterations = 4, MinimumExampleCountPerLeaf = 20, LearningRate = 0.4111824880638518, LabelColumnName = @"fare_amount", FeatureColumnName = @"Features", Booster = new GradientBooster.Options() { SubsampleFraction = 0.11834075080680864, FeatureFraction = 0.99999999, L1Regularization = 6.503478994394013E-10, L2Regularization = 0.23212120021727276 }, MaximumBinCountPerFeature = 253 }));
}
