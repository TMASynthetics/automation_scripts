# Triton.sh 
This script, `triton.sh`, is designed to automate the setup and deployment of the NVIDIA Triton Inference Server using Docker. It ensures that all necessary dependencies are installed, pulls the specified Triton Server docker image, and configures the environment using a predefined configuration.
This script has two modes of operation, cli mode and 

## Usage
This script 
### Dependencies:
  - docker.io
  - nvidia-container-toolkit
  - screen

### Triton_image
The Triton Server docker image we use in this project us:
  - nvcr.io/nvidia/tritonserver:25.02-py3

tritonconfigenv=".tritonconfigenv"
tritonconfigfile="https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/tritonconfig/tritonconfig.py"
tritonconfigrequirements="https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/tritonconfig/requirements.txt"

