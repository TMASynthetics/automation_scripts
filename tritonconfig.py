import tempfile, shutil, os, onnx
from config.models import MODELS
import subprocess

config = {
  "remove tmp files": False,
  "package models": True,
  "run tar command": True,
  "triton config file name": "config.pbtxt",
  "triton config max batch size": 5,
  "triton config platform": "onnxruntime_onnx",
  "triton model version number": 1,
}

def get_data_type_string(data_type):
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

def configen(model_file, outfile='/dev/stdout'):
    model_name = model_file.split(".")[0].split("/")[-1]
    platform = config["triton config platform"]
    batch_size = config["triton config max batch size"]

    with open(outfile, "w") as f:
        f.write(f"name: \"{model_name}\"\n")
        f.write(f"platform: \"{platform}\"\n")
        f.write(f"max_batch_size: {batch_size}\n")

        # Load the ONNX model
        model = onnx.load(model_file)
        graph = model.graph

        # Extract inputs
        f.write("\ninput [\n")
        for input_tensor in graph.input:
            f.write("\t{\n")
            name = input_tensor.name
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim][1:]
            data_type = input_tensor.type.tensor_type.elem_type
            readable_type = get_data_type_string(data_type)
            f.write(f"\t\tname: {name},\n\t\tdata_type: {readable_type},\n\t\tdims: {shape}\n")
            f.write("\t},\n")
        f.write("]\n")

        # Extract outputs
        f.write("\noutput [\n")
        for output_tensor in graph.output:
            f.write("\t{\n")
            name = output_tensor.name
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim][1:]
            data_type = output_tensor.type.tensor_type.elem_type
            readable_type = get_data_type_string(data_type)
            f.write(f"\t\tname: {name},\n\t\tdata_type: {readable_type},\n\t\tdims: {shape}\n")
            f.write("\t},\n")
        f.write("]\n")

def prepare(model, repo):
  path = model[1]["path"]
  model_name = os.path.basename(path).split(".")[0]
  print(f"Generating config for: {model_name}")

  # Create the directory for the model
  model_dir = os.path.join(repo, model_name)
  os.makedirs(model_dir, exist_ok=True)

  # Generate the config file
  config_path = os.path.join(model_dir, config["triton config file name"])
  configen(path, config_path)

  # Create the directory for the model version
  version_path = os.path.join(model_dir, str(config["triton model version number"]))
  os.makedirs(version_path)

  # Copy the model file
  model_path = os.path.join(version_path, "model.onnx")
  shutil.copy(path, model_path)

def pack_models(temp_dir):

  repo = os.path.join(temp_dir, "model_repository")
  os.makedirs(repo)

  for model in MODELS.items():
    if model is not None:
      try:
        prepare(model, repo)
      except Exception as e:
        print(f"An error occurred: {e}")

  if (config["run tar command"]):
    tar_path = os.path.join(temp_dir, "model_repository.tar.gz")
    print(f"Creating tarball of model repository at: {tar_path}")
    subprocess.run([
      "tar",
      "-czf",
      tar_path,
      "-C",
      temp_dir,
      "model_repository"],
      check=True
    )
    print(f"Model repository tarball created at: {tar_path}")

def main():
  temp_dir = tempfile.mkdtemp()  # Create the temp directory
  print(f"Temporary directory created at: {temp_dir}")

  if(config["package models"]):
    pack_models(temp_dir)

  if (config["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  main()