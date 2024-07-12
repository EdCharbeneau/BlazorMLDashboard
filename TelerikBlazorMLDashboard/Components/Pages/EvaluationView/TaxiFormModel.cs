using TelerikBlazorMLDashboard;
using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations;

namespace TelerikBlazorMLDashboard.Components.Pages.EvaluationView;

public class TaxiFormModel
{

    [Range(1, 6)]
    [Display(AutoGenerateField = false)]
    public int SelectedRateCode { get; set; } = 1;
    public RateCodeOption[] RateCodeOptions { get; set; } = [
        new(1, "Standard rate"),
        new(2, "JFK"),
        new(3, "Newark"),
        new(4, "Nassau or Westchester"),
        new(5, "Negotiated fare"),
        new(6, "Group ride")
        ];

    [Range(1, 6)]
    [Display(Name = "Passenger Count")]
    public int PassengerCount { get; set; } = 1;

    [Range(0.01, 100)]
    [Display(Name = "Trip Distance")]
    public float TripDistance { get; set; } = 1;

    [Display(AutoGenerateField = false)]
    public PaymentOption[] PaymentOptions { get; set; } = [
        new("CSH", "Cash"),
            new("CRD", "Credit")
        ];

    [Required]
    [Display(AutoGenerateField = false)]
    public string SelectedPayment { get; set; } = string.Empty;

    [Display(AutoGenerateField = false)]
    public VendorOption[] VendorOptions { get; set; } = [
        new("VTS", "Velocity Taxi Services"),
            new("CMT", "City Motion Taxis"),
        ];

    [Required]
    [Display(AutoGenerateField = false)]
    public string SelectedVendorOption { get; set; } = string.Empty;

    public record PaymentOption(string Value, string Name);
    public record VendorOption(string Value, string Name);
    public record RateCodeOption(int Value, string Name);

    public MLModel.TripModelInput ToTripModelInput() =>
        new MLModel.TripModelInput()
        {
            PassengerCount = PassengerCount,
            PaymentType = SelectedPayment,
            RateCode = SelectedRateCode,
            TripDistance = TripDistance,
            VendorId = SelectedVendorOption
        };

}

