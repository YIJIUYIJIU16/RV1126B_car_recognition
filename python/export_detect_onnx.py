import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../Car_recognition'))

from models.experimental import attempt_load

MODEL_DIR = '../model/'
MODEL_PATH = '../../../../Car_recognition/weights/detect.pt'
ONNX_MODEL_PATH = MODEL_DIR + 'detect.onnx'

def export_onnx():
    device = torch.device('cpu')
    
    model = attempt_load(MODEL_PATH, map_location=device)
    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid = [torch.zeros(1)] * 3
    model.model[-1].export_cat = True
    model.eval()
    
    img_size = 384
    dummy_input = torch.zeros(1, 3, img_size, img_size).to(device)
    
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
    
    y = model(dummy_input)
    
    print(f"Model output shape: {y.shape}")
    print(f"First sample values: {y[0, 0, :5]}")
    print(f"Confidence range: min={y[0, :, 4].min().item():.4f}, max={y[0, :, 4].max().item():.4f}")
    
    model.fuse()
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    if os.path.exists(ONNX_MODEL_PATH):
        print('ONNX model has been saved in ' + ONNX_MODEL_PATH)
    else:
        print('Export ONNX failed!')

if __name__ == '__main__':
    export_onnx()
