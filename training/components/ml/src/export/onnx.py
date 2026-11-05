import pickle
import torch
import torch.onnx as onnx
from export.factories.registry import get_op

def export(cfg, sktime_model_file_path: str, output_onnx_file_path: str, in_features: int):
    with open(sktime_model_file_path, 'rb') as f:
        clf = pickle.load(f)
    
    model = get_op(clf)
    
    input_names  = ['input']
    output_names = ['output']

    dummy_input = 10*torch.randn(1,in_features,85, dtype=torch.float)
    onnx.export(model,dummy_input,output_onnx_file_path,verbose=True,input_names=input_names,output_names=output_names, opset_version=11, dynamic_axes={'input':[0,2]})