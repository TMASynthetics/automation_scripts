import tempfile
import shutil
import os
import sys
import onnx
import subprocess
import argparse
import glob

MODELS= {}
Settings = {
  "package models": True,
  "remove tmp files": True,
  "run tar command": False
}
config = {
      "triton config file name": "config.pbtxt",
      "triton config max batch size": 0,
      "triton config platform": "onnxruntime_onnx",
      "triton model version number": 1,
    }
class TritonConfigBuilder:
  def __init__(self):
     self.config = config

  def get_data_type_string(self, data_type):
      type_map = {
          onnx.TensorProto.FLOAT: "TYPE_FP32",
          onnx.TensorProto.UINT8: "TYPE_UINT8",
          onnx.TensorProto.INT8: "TYPE_INT8",
          onnx.TensorProto.UINT16: "TYPE_UINT16",
          onnx.TensorProto.INT16: "TYPE_INT16",
          onnx.TensorProto.INT32: "TYPE_INT32",
          onnx.TensorProto.INT64: "TYPE_INT64",
          onnx.TensorProto.STRING: "TYPE_STRING",
          onnx.TensorProto.BOOL: "TYPE_BOOL",
          onnx.TensorProto.FLOAT16: "TYPE_FP16",
          onnx.TensorProto.DOUBLE: "TYPE_FP64",
          onnx.TensorProto.UINT32: "TYPE_UINT32",
          onnx.TensorProto.UINT64: "TYPE_UINT64",
      }
      return type_map.get(data_type, "UNKNOWN")  # Default to UNKNOWN if not found

  def configen(self, model_file, outfile='/dev/stdout'):
      model_name = model_file.split(".")[0].split("/")[-1]
      platform = self.config["triton config platform"]
      batch_size = self.config["triton config max batch size"]

      with open(outfile, "w") as f:
          # Load the ONNX model
          model = onnx.load(model_file)
          graph = model.graph

          f.write(f"name: \"{model_name}\"\n")
          f.write(f"platform: \"{platform}\"\n")
          f.write(f"max_batch_size: {batch_size}\n")

          length = len(graph.input)
          # Extract inputs
          f.write("\ninput [\n")
          for i, input_tensor in enumerate(graph.input):
              f.write("\t{\n")
              name = input_tensor.name
              shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in input_tensor.type.tensor_type.shape.dim]
              data_type = input_tensor.type.tensor_type.elem_type
              readable_type = self.get_data_type_string(data_type)
              f.write(f"\t\tname: \"{name}\",\n\t\tdata_type: {readable_type},\n\t\tdims: {shape}\n")
              f.write('\t},\n' if i != length - 1 else '\t}\n')
          f.write("]\n")

          length = len(graph.output)
          # Extract inputs
          f.write("\noutput [\n")
          for i, input_tensor in enumerate(graph.output):
              f.write("\t{\n")
              name = input_tensor.name
              shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
              data_type = input_tensor.type.tensor_type.elem_type
              readable_type = self.get_data_type_string(data_type)
              f.write(f"\t\tname: \"{name}\",\n\t\tdata_type: {readable_type},\n\t\tdims: {shape}\n")
              f.write('\t},\n' if i != length - 1 else '\t}\n')
          f.write("]\n")

  def prepare(self, model, repo):
    path = model[1]["path"]
    model_name = os.path.basename(path).split(".")[0]
    print(f"Generating config for: {model_name}")

    # Create the directory for the model
    model_dir = os.path.join(repo, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Generate the config file
    config_path = os.path.join(model_dir, self.config["triton config file name"])
    self.configen(path, config_path)

    # Create the directory for the model version
    version_path = os.path.join(model_dir, str(self.config["triton model version number"]))
    os.makedirs(version_path, exist_ok=True)

    # Copy the model file
    model_path = os.path.join(version_path, "model.onnx")
    shutil.copy(path, model_path, follow_symlinks=True, dirs_exist_ok=True)

  def pack_models(self, output_dir):
    repo = os.path.join(output_dir, "model_repository")
    os.makedirs(repo, exist_ok=True)
    for model in MODELS.items():
      if model is not None:
        try:
          self.prepare(model, repo)
        except Exception as e:
          print(f"An error occurred: {e}")
    if (Settings["run tar command"]):
      tar_path = os.path.join(output_dir, "model_repository.tar.gz")
      print(f"Creating tarball of model repository at: {tar_path}")
      subprocess.run([
        "tar",
        "-czf",
        tar_path,
        "-C",
        output_dir,
        "model_repository"],
        check=True
      )
      print(f"Model repository tarball created at: {tar_path}")

def run(args):
  if args.output is None and args.link is None:
    output_dir = tempfile.mkdtemp()  # Create the temp directory
    print(f"Temporary directory created at: {output_dir}")
  else:
    output_dir = args.output

  tritonbuilder = TritonConfigBuilder()
  if(Settings["package models"]):
    tritonbuilder.pack_models(output_dir)

  if (args.output is not None and args.output != output_dir):
    print(f"Copying {output_dir} dir to: {args.output}")
    shutil.copytree(output_dir, args.output, dirs_exist_ok=True)

  if (Settings["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(output_dir)

if __name__ == "__main__":
  args = argparse.ArgumentParser()
  args.description = "This script generates Triton configuration files for faceFatures"
  args.description += " models and packages them into a model repository.\n"
  args.description += "By default this script expects to find the models by their paths from the config/models.py file."
  args.description += "However, you can specify the path to the mmodels by passing the --models flag."
  args.epilog = "Example: python tritonconfig.py --output /path/to/output"
  args.add_argument("--output", help="Output directory", required=False)
  args.add_argument("--models", help="Path to the models", required=False)
  args = args.parse_args()

  if args.models is not None:
    sys.path.append(args.models)
    model_files = glob.glob(os.path.join(args.models, "*.onnx"))
    for model_file in model_files:
      model_name = os.path.basename(model_file).split(".")[0]
      MODELS[model_name] = {"path": model_file}
  else:
    from config.models import MODELS

  run(args=args)
