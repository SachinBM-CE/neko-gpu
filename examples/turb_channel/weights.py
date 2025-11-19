import torch
import os

model_path = '/home/sachinbm/post_processing/local/const_pres_grad/logdir/policy/model.pt'

def load_model(path):
    try:
        model = torch.jit.load(path)
        print("TorchScript model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    return model

# Load the model
model = load_model(model_path)

print('====================================================================')
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")
    print(f"{name}: {param.dtype}")
    print(param)
    print()

# Convert to float64
if model is not None:
    model_float64 = model.to(torch.float64)
    print("Model converted to float64")
    
    # Save the converted model
    output_path = '/home/sachinbm/post_processing/local/const_pres_grad/logdir/policy/model_float64.pt'
    torch.jit.save(model_float64, output_path)
    print(f"Float64 model saved to: {output_path}")
    
    # Verify the conversion
    print("\nVerifying conversion:")
    for name, param in model_float64.named_parameters():
        print(f"{name}: shape={param.shape}")
        print(f"{name}: {param.dtype}")
        print(param)
        print()

# import torch
# import torch.nn as nn
# import os

# # 1. DEFINE A NEW WRAPPER CLASS
# # This architecture is based on your sac_model.cpp and weights.py output.
# # It MUST have a 'forward' method.
# class StandalonePolicy(nn.Module):
#     def __init__(self):
#         super(StandalonePolicy, self).__init__()
#         # Re-create the layers from your C++ model
#         # Input (2) -> 64 (from shape [64, 2])
#         self.encoder_fc_0 = nn.Linear(2, 64)
#         # 64 -> 64 (from shape [64, 64])
#         self.encoder_fc_1 = nn.Linear(64, 64)
        
#         # The output layer for the action (mu)
#         # 64 -> 1 (from shape [1, 64])
#         self.out_fc_1_2 = nn.Linear(64, 1)
        
#         # We IGNORE the log_sigma layer (out_fc_2_2) because
#         # torchfort_inference only wants one output.
        
#         # Re-create the separate bias parameters
#         self.encoder_b_0 = nn.Parameter(torch.zeros(64))
#         self.encoder_b_1 = nn.Parameter(torch.zeros(64))
#         self.out_b_1_2 = nn.Parameter(torch.zeros(1))
        
#         # We also ignore out_b_2_2 (for log_sigma)

#     # 2. THIS IS THE 'forward' METHOD 'torchfort_inference' IS LOOKING FOR
#     def forward(self, state):
#         # ... (layer logic from sac_model.cpp) ...
#         x = state.reshape((state.size(0), -1)) 
#         x = torch.relu(self.encoder_fc_0(x) + self.encoder_b_0)
#         x = torch.relu(self.encoder_fc_1(x) + self.encoder_b_1)
        
#         # This replicates C++ line 122:
#         # We get 'mu' (the first output)
#         mu = self.out_fc_1_2(x) + self.out_b_1_2
        
#         # This replicates C++ line 125:
#         # We apply the 'tanh' squashing
#         action = torch.tanh(mu)
        
#         return action

# # --- Main Script Logic ---

# print("Loading original float32 model...")
# model_path = '/home/sachinbm/post_processing/local/const_pres_grad/logdir/policy/model.pt'
# output_path = '/home/sachinbm/post_processing/local/const_pres_grad/logdir/policy/model_float64.pt'

# # Load the original model (which is missing 'forward')
# try:
#     original_model = torch.jit.load(model_path)
# except Exception as e:
#     print(f"Failed to load original model: {e}")
#     exit()

# # Get its weights
# original_weights = original_model.state_dict()

# print("Creating new wrapper model...")
# # Create an instance of our new class
# export_model = StandalonePolicy()

# # Load the weights into the new model
# # We use strict=False because our new model is intentionally
# # missing the log_sigma layers (out_fc_2_2, out_b_2_2)
# export_model.load_state_dict(original_weights, strict=False)
# export_model.eval() # Set to evaluation mode

# print("Scripting new model...")
# # Create an example input to trace the model
# # Input shape is [batch_size, state_dim] = [1, 2]
# example_input = torch.randn(1, 2)

# # Script the new model. This will trace the 'forward' method.
# scripted_model = torch.jit.trace(export_model, example_input)

# print("Converting to float64 and saving...")
# # Convert to float64
# scripted_model.to(torch.float64)

# # Save the final file
# scripted_model.save(output_path)

# print(f"New model with 'forward' method saved to: {output_path}")

# # --- Verification Step ---
# print("\nVerifying new model...")
# try:
#     verify_model = torch.jit.load(output_path)
#     print("New model loaded successfully.")
    
#     # Try running the forward method
#     test_input = torch.randn(1, 2, dtype=torch.float64)
#     output = verify_model.forward(test_input)
#     print(f"'forward' method executed successfully. Output: {output}")
#     print("\nParameters in new model (all should be float64):")
#     for name, param in verify_model.named_parameters():
#         print(f"{name}: {param.dtype}")

# except Exception as e:
#     print(f"\nVerification failed: {e}")