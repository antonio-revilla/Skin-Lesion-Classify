import torch.onnx
from models import NN_Model


model_name = 'resnet50'
model_path = f'trained_models/{model_name}.pth'

model = NN_Model(model_name)
model.load_state_dict(torch.load(model_path))
model.eval()

dummy_input = torch.randn(1, 3, model.input_size, model.input_size)
torch.onnx.export(model, dummy_input, f'trained_models/{model_name}.onnx', verbose=True)