namespace BlazorMLDashboard;
public partial class MLModel
{
    public class TestDataResults
    {
        public TestDataResults() { }
        public TestDataResults(IEnumerable<TestDataPoint> resultSet) => ResultSet = resultSet;
        public IEnumerable<TestDataPoint> ResultSet { get; set; } = [];
        public IEnumerable<DataPoint> MinimizedSquareError => GetMinimizedSquareError();
        public RegressionMetrics RegressionMetrics { get; set; } = new();
        public List<RegressionMetrics>? PermutationFeatureImportance { get; set; }

        private IEnumerable<DataPoint> GetMinimizedSquareError()
        {
            var funcY = GetRegressionFunction();
            var min = ResultSet.Min(x => x.Actual);
            var max = ResultSet.Max(x => x.Actual);
            var a = new DataPoint { X = min, Y = funcY(min) };
            var b = new DataPoint { X = max, Y = funcY(max) };
            return new[] { a, b };
        }

        private Func<double, double> GetRegressionFunction()
        {
            // Regression Line calculation explanation:
            // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

            double xyMultiTotal = ResultSet.Sum(r => r.Actual * r.Predicted);
            double xSquareTotal = ResultSet.Sum(r => r.Actual * r.Actual);

            double meanX = ResultSet.Average(r => r.Actual);
            double meanY = ResultSet.Average(r => r.Predicted);

            double meanXY = ResultSet.Average(r => r.Actual * r.Predicted);
            double meanXsquare = ResultSet.Average(r => r.Actual * r.Actual);

            double mslope = ((meanX * meanY) - meanXY) / ((meanX * meanX) - meanXsquare);

            double bintercept = meanY - (mslope * meanX);

            //Generic function for Y for the regression line
            // y = (m * x) + b;

            ////Function for Y1 in the line
            return (double x) => (mslope * x) + bintercept;
        }
    }

    public class RegressionMetrics
    {
        public string? Feature { get; set; }
        public double RSquared { get; set; }
        public double RootMeansSquaredError { get; set; }
        public double MeanSquaredError { get; set; }
        public double MeanAbsoluteError { get; set; }
    }
}

