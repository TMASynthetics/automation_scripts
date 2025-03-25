# automation_scripts
Here is a collection of tools for setting up and tunning a local testing environment for Nvidia's Triton Inference Server.

## Contents of this repository:

> [**tritonconfig**](/tritonconfig) is used for automatically generating config.pbtxt files and packaging them along with your models into a model_repository and/or a model_repository.tar.gz file ready for the Triton inference server to use

> [**tritonserver**](/titonserver) is a tool used to build and manage the nvidia Triton inference sevrer. It has a helpful TUI interface.

> [**buildlayer**](/buildlayer) *EXPERIMENTAL* is used for building out our lambda_function.py files based of a template


## How to use

Once you have cloned a local copy of this repository you will find the tools organised into sub-directories, each directory contains a README and all the items necessary to run.

At the root of this repository is a file named `install.sh` it allows you to access any of the scripts from any directory by either adding them to the $PATH environment variable or by creating a symlink to /usr/bin

## What is Triton Inference Server?

Triton Inference Server is an open-source software solution developed by NVIDIA to streamline AI inferencing. 

Triton Inference Server acts as a bridge between your machine learning models and the applications that use them. It takes requests from applications, feeds the data to the models and returns the results.

It supports a wide range of machine learning and deep learning frameworks, including TensorFlow, PyTorch, TensorRT, ONNX, and more. Triton is the inference engine that we use at TMA for our cloud inferencing tasks. So it's important that we develop our applications with Triton as our AI backend.

### Examples of a Triton testing environment

```
                            Computer 1                     
                            with the application           
        +----------------------------------------+         
        |   Application                          |         
        |                                        |         
        |  (Sends API        (Recieves the data  |         
        |   Requests)        via the API)        |         
        +----------------------------------------+         
          |                       ^                        
          |                       |      Computer 2        
          v                       |  with Nvidia GPU's     
+---------------------------------------------------------+
|  Triton Inference Server                                |
|   (Performs inference tasks with the requested model)   |
|        +-------------------+  +-------------------+     |
|        |  Model 1          |  |  Model 2          |     |
|        |  (e.g., Image     |  |  (e.g., face      |     |
|        |   Recognition)    |  |   enhancemnet)    |     |
|        +-------------------+  +-------------------+     |
|                                                         |
+---------------------------------------------------------+
```

* The Application sends API requests to the Triton Inference Server.
    Specifying which model they want to use
* The Triton Inference Server routes the requests to the appropriate Models.
* The Models process the data and generate results.
* The Triton Inference Server sends the results back to the Application.

This method is benefitial for Cloud AI applications, it's dynamic and resilient. Triton can combine multiple inference requests into a single batch while load balancing the inference across available GPU's, enhancing throughput and minimizing latency. Triton can be integrated into Kubernetes for managing and scaling AI applications, making it cloud-native and suitable for dynamic environments and is well supported by Cloud provider platforms like Google Cloud and AWS.

### Setting up Triton Server locally
You may wish to set up Triton on the same machine that your application is running from or you may wish to host Triton Server on a separate machine from your application. Either way works well.

STEPS:
#### 1. CUDA Drivers
You will need to have an Nvidia CUDA GPU and have the drivers needed for CUDA already installed on the machine where Triton Server will run.

Since each system and GPU may require very different steps to installing the right drivers we haven't included a function for doing this automatically at this time.

Please See https://www.nvidia.com/drivers to help you get started

#### 2. Building model_repository
> Triton uses a model_repository for acquiring it's models, in most cases you only need to download [triton.sh](/titonserver/triton.sh) as it serves as a wrapper for the other automation_scripts. Triton.sh uses [tritonconfig](/tritonconfig/) which is a tool for building the model_repository and configuration files needed for Triton Server. If you need to build the model_repository on a separate machine from your instance of Triton Server, you can use this tool to generate the model_repository and manually move it to the machine which will host Triton Server. 

**Start the triton.sh script**, **select Option 2** from the menu. The script will try to search the current working directory for any Face-Features project files and will ask you to **select which project you want to build the model_repository for**. You can now choose a name for the directory which will be placed at the root of the selected project and will contain the model_repository.

#### 3. Run the server
Start the triton.sh script, select option 3 in the menu, when prompted select the model_repository that you want to use in Triton Server and Triton Server will be started in the background. The script automatically performs checks to make sure Triton Server has started correctly.

##### TIPS:
* When building model_repositories triton.sh will try to search for Face-Features projects in any subdirectories of the current working directory up to 5 layers deep.
* Also when starting Triton Server triton.sh will try to search for any model_repositories in any subdirectories of the current working directory also up to 5 layers deep

  - By adding the /path/to/automation_scripts/tritonserver to the $PATH environment variable you can simply navigate to any project directory and call triton.sh it's a very handy way to start triton.sh from any directory you need.
