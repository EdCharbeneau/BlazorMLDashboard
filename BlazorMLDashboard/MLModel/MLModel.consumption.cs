using Microsoft.ML;
using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations;
namespace BlazorMLDashboard;
public partial class MLModel
{
    /// <summary>
    /// model input class for MLModel.
    /// </summary>
    #region model input class
    public class TripModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"vendor_id")]
        public string VendorId { get; set; } = string.Empty;

        [LoadColumn(1)]
        [ColumnName(@"rate_code")]
        public float RateCode { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"passenger_count")]
        public float PassengerCount { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"trip_distance")]
        public float TripDistance { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"payment_type")]
        public string PaymentType { get; set; } = string.Empty;

        [LoadColumn(6)]
        [ColumnName(@"fare_amount")]
        [Display(AutoGenerateField = false)]
        public float FareAmount { get; set; }

    }

    #endregion

    /// <summary>
    /// model output class for MLModel.
    /// </summary>
    #region model output class
    public class TripModelOutput
    {
        [ColumnName(@"vendor_id")]
        public float[]? VendorId { get; set; }

        [ColumnName(@"rate_code")]
        public float RateCode { get; set; }

        [ColumnName(@"passenger_count")]
        public float PassengerCount { get; set; }

        [ColumnName(@"trip_distance")]
        public float TripDistance { get; set; }

        [ColumnName(@"payment_type")]
        public float[]? PaymentType { get; set; }

        [ColumnName(@"fareAmount")]
        public float FareAmount { get; set; }

        [ColumnName(@"Features")]
        public float[]? Features { get; set; }

        [ColumnName(@"Score")]
        public float Score { get; set; }

    }

    #endregion
    public class TestDataPoint
    {
        [ColumnName("fare_amount"), LoadColumn(6)]
        public float Actual { get; set; }
        [ColumnName("Score")]
        public float Predicted { get; set; }
    }

    private ModelSettings Settings => options.Value;
    public bool IsModelCreated => System.IO.File.Exists(Settings.GetPrivatePath(Settings.ModelFileName));

    public Lazy<PredictionEngine<TripModelInput, TripModelOutput>> PredictEngine => new Lazy<PredictionEngine<TripModelInput, TripModelOutput>>(() => CreatePredictEngine(), true);

    private PredictionEngine<TripModelInput, TripModelOutput> CreatePredictEngine()
    {
        var mlContext = new MLContext();
        ModelSettings settings = options.Value;
        ITransformer mlModel = mlContext.Model.Load(settings.GetPrivatePath(settings.ModelFileName), out var _);
        return mlContext.Model.CreatePredictionEngine<TripModelInput, TripModelOutput>(mlModel);
    }

    /// <summary>
    /// Use this method to predict on <see cref="TripModelInput"/>.
    /// </summary>
    /// <param name="input">model input.</param>
    /// <returns><seealso cref=" TripModelOutput"/></returns>
    public TripModelOutput Predict(TripModelInput input)
    {
        var predEngine = PredictEngine.Value;
        return predEngine.Predict(input);
    }
}

