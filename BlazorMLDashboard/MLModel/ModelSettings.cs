namespace BlazorMLDashboard;
public class ModelSettings
{
    public required string RetrainFileName { get; set; }
    public required string EvaluationFileName { get; set; }
    public required string ModelFileName { get; set; }
    public required string PreprocessedTrainDataFileName { get; set; }
    public required string AnalysisFileName { get; set; }
    public required string PublicPath { get; set; }
    public required string PrivatePath { get; set; }
    public char RetrainSeparatorChar { get; set; }
    public bool RetrainHasHeader { get; set; }
    public bool RetrainAllowQuoting { get; set; }

    public string GetPrivatePath(string fileName) => Path.Combine(Environment.CurrentDirectory, PrivatePath, fileName);
    public string GetPublicPath(string fileName) => Path.Combine(Environment.CurrentDirectory, PublicPath, fileName);
}