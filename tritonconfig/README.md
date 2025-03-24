# Tritonconfig
This script generates Triton configuration files for ONNX models and packages them into a model_repository.

Unless the `--models MODELS_DIR` flag is set this script uses the `config/models.py` file from each of the Face-Features projects to find the relevent models. So this file either must be placed at the root of any project you wish to use in Triton Server or called with the `--models` flag.

Classes:
  TritonConfigBuilder: A class to build Triton configuration files for ONNX models.

Functions:
  run(outputdir=None): Main function to generate Triton configuration files and package models.

Usage:
  Example: python tritonconfig.py --output /path/to/output --models /path/to/models

Arguments:
  * `--output`: Output directory where the model repository will be saved.
  * `--models`: Path to the directory containing ONNX models. If not provided, models will be imported from config/models.py.

Settings python dict:
  package models: Boolean flag to indicate whether to package models.
  remove tmp files: Boolean flag to indicate whether to remove temporary files after execution.

Config python dict:
  * run tar command: Boolean flag to indicate whether to create a tarball of the model repository.
  * triton config file name: Name of the Triton configuration file.
  * triton config max batch size: Maximum batch size for Triton configuration.
  * triton config platform: Platform for Triton configuration.
  * triton model version number: Version number for the Triton model.

Methods in TritonConfigBuilder:
  __init__: Initializes the TritonConfigBuilder with default configuration.
  get_data_type_string(data_type): Returns the Triton data type string for a given ONNX data type.
  configen(model_file, outfile='/dev/stdout'): Generates a Triton configuration file for a given ONNX model.
  prepare(model, repo): Prepares the model repository by generating configuration files and copying model files.
  pack_models(temp_dir): Packages the models into a model repository and optionally creates a tarball.

Exceptions:
  Handles exceptions during model preparation and logs the error message.