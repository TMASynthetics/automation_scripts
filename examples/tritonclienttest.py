import tritonclient.http as httpclient
import numpy as np

TRITON_SERVER_URL = "host.local:8000"
MODEL_NAME = "2dfan4"

# Terminal Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Create a client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Check if the server is live
if client.is_server_live():
    print("Triton server is live!")
else:
    print("{RED}Triton server is down!{RESET}")
    exit(1)

# Check if the server is ready
if client.is_server_ready():
    print("Triton server is ready to accept requests!")
else:
    print("{RED}Triton server is not ready!{RESET}")
    exit(1)

# Generate test input data (random noise just to check processing)
input_data = np.random.rand(1, 3, 256, 256).astype(np.float32)

# Prepare Triton input
input_tensor = httpclient.InferInput("input", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

print(f"{YELLOW}⚡ Sending inference request to Triton...{RESET}")

# Send inference request
response = client.infer(model_name=MODEL_NAME, inputs=[input_tensor])

# Extract outputs
landmarks = response.as_numpy("landmarks")
heatmaps = response.as_numpy("heatmaps")

# Validate output shapes
assert landmarks.shape == (1, 68, 3), f"Unexpected landmarks shape: {landmarks.shape}"
assert heatmaps.shape == (1, 68, 64, 64), f"Unexpected heatmaps shape: {heatmaps.shape}"

# Print sample outputs
print(f"{GREEN}🔥 Triton Inference Test Passed!{RESET}")
print(f"{CYAN}Landmarks sample:{RESET}\n{landmarks[0, :5]}")  # Print first 5 landmarks
print(f"{CYAN}Heatmaps shape: {RESET}{heatmaps.shape}")

print(f"""{BLUE}
We need to create InferInput objects for each input tensor of the model.
Then we can set the data of the input tensors using the set_data_from_numpy method.
Finally, we can send the request to the server using the infer method.

input1 = httpclient.InferInput("input", input1_data.shape, "FP32")
input2 = httpclient.InferInput("input", input2_data.shape, "FP32")

input1.set_data_from_numpy(input1_data)
input2.set_data_from_numpy(input2_data)

response = client.infer(model_name="model_name", inputs=[input1, input2])
{RESET}""")