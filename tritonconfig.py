import tempfile, shutil, os, sys, onnx
from config.models import MODELS
import subprocess

Settings = {
  "package models": True,
  "remove tmp files": False,
}
config = {
      "run tar command": False,
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
          f.write(f"name: \"{model_name}\"\n")
          f.write(f"platform: \"{platform}\"\n")
          f.write(f"max_batch_size: {batch_size}\n")
          
          # Load the ONNX model
          model = onnx.load(model_file)
          graph = model.graph

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
    os.makedirs(version_path)

    # Copy the model file
    model_path = os.path.join(version_path, "model.onnx")
    shutil.copy(path, model_path)

  def pack_models(self, temp_dir):
    repo = os.path.join(temp_dir, "model_repository")
    os.makedirs(repo)
    for model in MODELS.items():
      if model is not None:
        try:
          self.prepare(model, repo)
        except Exception as e:
          print(f"An error occurred: {e}")
    if (self.config["run tar command"]):
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

def run(outputdir=None):
  temp_dir = tempfile.mkdtemp()  # Create the temp directory
  print(f"Temporary directory created at: {temp_dir}")

  tritonbuilder = TritonConfigBuilder()
  if(Settings["package models"]):
    tritonbuilder.pack_models(temp_dir)

  if (outputdir is not None):
    print(f"Copying {temp_dir} dir to: {outputdir}")
    shutil.copytree(temp_dir, outputdir, dirs_exist_ok=True)

  if (Settings["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  outputdir = None
  print(sys.argv[2])
  if (len(sys.argv)) > 2:
    if (sys.argv[1] == "--output"):
       outputdir = sys.argv[2]
  run(outputdir=outputdir)
