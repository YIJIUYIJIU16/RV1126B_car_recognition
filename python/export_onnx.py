import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../Car_recognition'))

from car_recognition.myNet import myNet

MODEL_DIR = '../model/'
MODEL_PATH = '../../../../Car_recognition/weights/car_rec_color.pth'
ONNX_MODEL_PATH = MODEL_DIR + 'car_rec_color.onnx'

def export_onnx():
    device = torch.device('cpu')
    
    check_point = torch.load(MODEL_PATH, map_location=device)
    cfg = check_point['cfg']
    num_classes = 11
    
    model = myNet(num_classes=num_classes, cfg=cfg)
    model.load_state_dict(check_point['state_dict'])
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    
    if os.path.exists(ONNX_MODEL_PATH):
        print('ONNX model has been saved in ' + ONNX_MODEL_PATH)
    else:
        print('Export ONNX failed!')

if __name__ == '__main__':
    export_onnx()
