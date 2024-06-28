using Microsoft.ML;
using System.Text.Json;

namespace BlazorMLDashboard;

public partial class MLModel
{
    public void Evaluate()
    {
        var mlContext = new MLContext();

        IDataView testData = LoadIDataViewFromFile(mlContext, Settings);
        ITransformer trainedModel = mlContext.Model.Load(Settings.GetPrivatePath(Settings.ModelFileName), out var _);
        var preprocessedTrainData = trainedModel.Transform(testData);
        using (FileStream stream = new FileStream(Settings.GetPublicPath(Settings.PreprocessedTrainDataFileName), FileMode.Create))
            mlContext.Data.SaveAsText(preprocessedTrainData, stream, Settings.RetrainSeparatorChar, true, false, false);
        //var pfi = CalculatePFI(mlContext, preprocessedTrainData, trainedModel, @"fare_amount");
        //var text = System.Text.Json.JsonSerializer.Serialize(pfi);
        //System.IO.File.WriteAllText(Path.Combine(StatsPath, "pmi.json"),text);
        System.IO.File.WriteAllText(Settings.GetPublicPath(Settings.AnalysisFileName), CalculateOverallStats(mlContext, preprocessedTrainData));
    }

    /// <summary>
    /// Permutation feature importance (PFI) is a technique to determine the importance 
    /// of features in a trained machine learning model. PFI works by taking a labeled dataset, 
    /// choosing a feature, and permuting the values for that feature across all the examples, 
    /// so that each example now has a random value for the feature and the original values for all other features.
    /// The evaluation metric (e.g. R-squared) is then calculated for this modified dataset, 
    /// and the change in the evaluation metric from the original dataset is computed. 
    /// The larger the change in the evaluation metric, the more important the feature is to the model.
    /// 
    /// PFI typically takes a long time to compute, as the evaluation metric is calculated 
    /// many times to determine the importance of each feature. 
    /// 
    /// </summary>
    /// <param name="mlContext">The common context for all ML.NET operations.</param>
    /// <param name="trainData">IDataView used to evaluate the model.</param>
    /// <param name="model">Model to evaluate.</param>
    /// <param name="labelColumnName">Label column being predicted.</param>
    /// <returns>A list of each feature and its importance.</returns>
    public List<Tuple<string, double>> CalculatePFI(MLContext mlContext, IDataView preprocessedTrainData, ITransformer model, string labelColumnName)
    {

        var permutationFeatureImportance =
     mlContext.Regression
     .PermutationFeatureImportance(
             model,
             preprocessedTrainData,
             labelColumnName: labelColumnName);

        var featureImportanceMetrics =
             permutationFeatureImportance
             .Select((kvp) => new { kvp.Key, kvp.Value.RSquared })
             .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean));

        var featurePFI = new List<Tuple<string, double>>();
        foreach (var feature in featureImportanceMetrics)
        {
            var pfiValue = Math.Abs(feature.RSquared.Mean);
            featurePFI.Add(new Tuple<string, double>(feature.Key, pfiValue));
        }

        return featurePFI;
    }

    public string CalculateOverallStats(MLContext mlContext, IDataView preprocessedTrainData)
    {
        var data = mlContext.Data
               .CreateEnumerable<TestDataPoint>(preprocessedTrainData, false)
               .Take(1000);
        TestDataResults result = new TestDataResults(data);
        var metrics = mlContext.Regression.Evaluate(preprocessedTrainData, "fare_amount", "Score");

        // Save to JSON
        result.RSquared = metrics.RSquared;
        result.RootMeansSquaredError = metrics.RootMeanSquaredError;
        result.MeanSquaredError = metrics.MeanSquaredError;
        result.MeanAbsoluteError = metrics.MeanAbsoluteError;
        return System.Text.Json.JsonSerializer.Serialize(result, new JsonSerializerOptions(JsonSerializerDefaults.Web));
    }
}