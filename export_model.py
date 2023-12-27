import torch.onnx
from policy_critic_network import policy_critic_network
# load the trained model from ./models/default.model
model = policy_critic_network(300, 124)
model.load_state_dict(torch.load("./models/default.model"))


print(model)
torch.onnx.export(model,               # model being run
                  torch.randn(1, 300), # model input (or a tuple for multiple inputs)
                  "/home/cs20btech11024/onnx/compiler_gym_ir2vec.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,input_names=["obs"])      # store the trained parameter weights inside the model file``