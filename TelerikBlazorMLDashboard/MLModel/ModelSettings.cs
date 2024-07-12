namespace TelerikBlazorMLDashboard;
public class ModelSettings
{
    public required string RetrainFileName { get; set; }
    public required string EvaluationFileName { get; set; }
    public required string ModelFileName { get; set; }
    public required string PreprocessedTrainDataFileName { get; set; }
    public required string AnalysisFileName { get; set; }
    public required string PMIFileName { get; set; }
    public required string DataPath { get; set; }
    public char RetrainSeparatorChar { get; set; }
    public bool RetrainHasHeader { get; set; }
    public bool RetrainAllowQuoting { get; set; }

    public string GetPath(string fileName) => Path.Combine(Environment.CurrentDirectory, DataPath, fileName);
}