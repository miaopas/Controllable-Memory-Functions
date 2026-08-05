"""Test script for result_processing utilities."""

from src.utils.result_processing import get_project_data, generate_result_pdf, MODEL_KEY_REGISTRY

# Test with LSTM_impulse_experimental project
project_name = "LSTM_impulse_experimental"
model_name = "LSTM"

print("=" * 60)
print("Testing result_processing module")
print(f"Project: {project_name}")
print("=" * 60)

print("\nAvailable models in registry:")
for model, key in MODEL_KEY_REGISTRY.items():
    print(f"  - {model}: {key}")

# Step 1: Get and process data
print("\n1. Retrieving and processing project data...")
df = get_project_data(project_name, results_dir="results")
print(f"   Retrieved {len(df)} runs")
print(f"   DataFrame shape: {df.shape}")
print(f"   Columns available: {list(df.columns)[:10]}...")  # Show first 10 columns

# Step 2: Generate PDF report (uses MODEL_KEY_REGISTRY automatically)
print("\n2. Generating PDF report...")
pdf_path = generate_result_pdf(
    project_name=project_name,
    model_name=model_name,
    results_dir="results",
    threshold=1e-2,  # Adjusted threshold for impulse experiments
)
print(f"   PDF saved to: {pdf_path}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
