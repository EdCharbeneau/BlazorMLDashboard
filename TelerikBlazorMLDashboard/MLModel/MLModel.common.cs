using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace BlazorMLDashboard;

public partial class MLModel(IOptions<ModelSettings> options)
{
    public ModelSettings Settings => options.Value;

    /// <summary>
    /// Load an IDataView from a file path.
    /// </summary>
    /// <param name="mlContext">The common context for all ML.NET operations.</param>
    /// <param name="modelSettings">Settings for ML.NET operations.</param>
    /// <returns>IDataView with loaded training data.</returns>
    public IDataView LoadIDataViewFromFile(MLContext mlContext, string fileName, ModelSettings modelSettings) =>
        mlContext.Data.LoadFromTextFile<TripModelInput>(modelSettings.GetPath(modelSettings.RetrainFileName), modelSettings.RetrainSeparatorChar, modelSettings.RetrainHasHeader, allowQuoting: modelSettings.RetrainAllowQuoting);
    public bool IsModelCreated => System.IO.File.Exists(Settings.GetPath(Settings.ModelFileName));

}
