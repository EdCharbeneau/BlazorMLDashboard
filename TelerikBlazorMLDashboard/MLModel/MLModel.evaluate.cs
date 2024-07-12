using Microsoft.ML;
using System.Text.Json;

namespace TelerikBlazorMLDashboard;

public partial class MLModel
{
    public void Evaluate()
    {
        MLContext mlContext = new();

        //Load Evaluation Data
        IDataView testData = LoadIDataViewFromFile(mlContext, Settings.EvaluationFileName, Settings);

        //Load the previously trained model under evaluation
        ITransformer trainedModelUnderEval = mlContext.Model.Load(Settings.GetPath(Settings.ModelFileName), out var _);

        //Using the trained model, evaluate the data set
        IDataView preprocessedTrainData = ProcessAndSaveScores(mlContext, testData, trainedModelUnderEval);

        CalculateAndSaveOverallStats(mlContext, trainedModelUnderEval, preprocessedTrainData);

    }

    private void CalculateAndSaveOverallStats(MLContext mlContext, ITransformer trainedModel, IDataView preprocessedTrainData)
    {
        TestDataResults analysisResults = CalculateOverallStats(mlContext, preprocessedTrainData);
        analysisResults.PermutationFeatureImportance = CalculatePFI(mlContext, preprocessedTrainData, trainedModel, @"fare_amount");
        string analysisResultsJson = JsonSerializer.Serialize(analysisResults, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        System.IO.File.WriteAllText(Settings.GetPath(Settings.AnalysisFileName), analysisResultsJson);
    }

    private IDataView ProcessAndSaveScores(MLContext mlContext, IDataView testData, ITransformer trainedModelUnderEval)
    {
        var preprocessedTrainData = trainedModelUnderEval.Transform(testData);

        //Save the processed data
        using (FileStream stream = new FileStream(Settings.GetPath(Settings.PreprocessedTrainDataFileName), FileMode.Create))
            mlContext.Data.SaveAsText(preprocessedTrainData, stream, Settings.RetrainSeparatorChar, true, false, false);
        return preprocessedTrainData;
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
    public List<RegressionMetrics> CalculatePFI(MLContext mlContext, IDataView preprocessedTrainData, ITransformer model, string labelColumnName)
    {

        var permutationFeatureImportance =
         mlContext.Regression
            .PermutationFeatureImportance(
                 model,
                 preprocessedTrainData,
                 labelColumnName: labelColumnName);

        return permutationFeatureImportance
             //.Select((kvp) => new { kvp.Key, kvp.Value })
             .OrderByDescending(myFeatures => Math.Abs(myFeatures.Value.RSquared.Mean))
             .Select(feature => new RegressionMetrics
             {
                 Feature = feature.Key,
                 MeanAbsoluteError = feature.Value.MeanAbsoluteError.Mean,
                 MeanSquaredError = feature.Value.MeanSquaredError.Mean,
                 RootMeansSquaredError = feature.Value.RootMeanSquaredError.Mean,
                 RSquared = Math.Abs(feature.Value.RSquared.Mean)
             }).ToList();
    }

    public TestDataResults CalculateOverallStats(MLContext mlContext, IDataView preprocessedTrainData)
    {
        IEnumerable<TestDataPoint> data = mlContext.Data
               .CreateEnumerable<TestDataPoint>(preprocessedTrainData, false)
               .Take(1000);

        TestDataResults result = new TestDataResults(data);
        Microsoft.ML.Data.RegressionMetrics metrics = mlContext.Regression.Evaluate(preprocessedTrainData, @"fare_amount", "Score");

        result.RegressionMetrics.RSquared = metrics.RSquared;
        result.RegressionMetrics.RootMeansSquaredError = metrics.RootMeanSquaredError;
        result.RegressionMetrics.MeanSquaredError = metrics.MeanSquaredError;
        result.RegressionMetrics.MeanAbsoluteError = metrics.MeanAbsoluteError;
        return result;
    }
}