import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from src.models import get_model
from src.engine import quantize_model

# --- LOAD DATASET FROM HUGGING FACE ---
# Replace with your actual username/dataset name
DATASET_NAME = "aayushkrm/wunder-fund-hft-data" 
try:
    print("Loading dataset from Hugging Face...")
    # Load first 1% just for demo speed
    dataset = load_dataset(DATASET_NAME, split="train[:1%]")
    df = dataset.to_pandas()
    # Ensure parquet structure matches
    SEQ_IDS = df['seq_ix'].unique()
except Exception as e:
    print(f"Could not load HF dataset: {e}")
    # Fallback to dummy data
    df = None
    SEQ_IDS = [0, 1, 2]

# --- CACHED MODEL LOADER ---
# In a real app, you would upload your 'best_model.pt' to the HF Space
# Here we initialize a random model to demonstrate the ARCHITECTURE if weights aren't found.
def load_cached_model():
    model = get_model("winner", hidden_size=240, layers=6)
    # Simulate quantization
    model = quantize_model(model)
    return model

MODEL = load_cached_model()

def inference(seq_id, steps_to_plot):
    if df is not None:
        # Extract sequence
        seq_data = df[df['seq_ix'] == seq_id].sort_values('step_in_seq')
        # Get raw features (0-31)
        raw_values = seq_data[[str(i) for i in range(32)]].values.astype(np.float32)
        # Normalize simple
        mean = raw_values.mean(axis=0)
        std = raw_values.std(axis=0) + 1e-6
        norm_values = (raw_values - mean) / std
    else:
        # Dummy data
        norm_values = np.random.randn(1000, 32).astype(np.float32)

    # Run Inference
    x = torch.tensor(norm_values).unsqueeze(0) # (1, 1000, 32)
    
    with torch.no_grad():
        # Get hidden states / predictions
        # Note: The model returns (prediction, hidden), but prediction is next step
        # We want to visualize the flow.
        preds = []
        h = None
        # Slow loop for demo visualization
        for t in range(min(len(x[0]), steps_to_plot)):
            xt = x[:, t:t+1, :]
            o, h = MODEL(xt, h)
            preds.append(o.numpy()[0,0,0]) # Plot 1st feature dim
            
    # Plotting
    fig = go.Figure()
    # Plot actual Feature 0
    fig.add_trace(go.Scatter(y=norm_values[:steps_to_plot, 0], mode='lines', name='Actual Feature 0', line=dict(color='gray')))
    # Plot predicted Feature 0
    fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted Feature 0', line=dict(color='green')))
    
    fig.update_layout(title=f"Forecasting Sequence {seq_id}", xaxis_title="Time Step", yaxis_title="Normalized Value")
    
    return fig

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# ⚡ Quant-Lab: HFT Sequence Modeling")
    gr.Markdown(f"**Strategy ED (Rank 28):** SE-Mish-DeepResGRU (INT8 Quantized)")
    
    with gr.Row():
        seq_selector = gr.Dropdown(choices=list(SEQ_IDS[:20]), label="Select Market Sequence", value=SEQ_IDS[0])
        step_slider = gr.Slider(minimum=50, maximum=1000, value=200, label="Steps to Visualize")
    
    plot = gr.Plot(label="Forecast Visualization")
    
    btn = gr.Button("Run Inference")
    btn.click(inference, inputs=[seq_selector, step_slider], outputs=plot)

if __name__ == "__main__":
    demo.launch()