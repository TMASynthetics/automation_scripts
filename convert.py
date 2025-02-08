import tempfile, shutil, os, onnx, ast
from config.models import MODELS
import subprocess

config = {
  "remove tmp files": True,
  "template file": "/home/josj/scr/MEPS/automation_scripts/templates/lambda_function",
  "compiled file": "lambda_function.py",
  "package models": True,
  "run tar command": False,
  "triton config file name": "config.pbtxt",
  "triton config max batch size": 5,
  "triton config platform": "onnxruntime_onnx",
  "triton model version number": 1,
  "Final project location directory": "aws_lambda"
}

def inject_lines(lines, fileHandle):
  for line in lines:
    fileHandle.write(f"{line}\n")

def inject_with_indent(lines, fileHandle, indent):
  # Write each line with a dynamic level of indentation
  if lines is not None:
    lines = [line.strip(" ") for line in lines]
    lines = [line.strip("\t") for line in lines]
    tabs = ' '*indent*2
    for line in lines:
      fileHandle.write(f"{tabs}{line}\n")

def build_from_template(template_file, output_file, imports, types, class_code, pipeline):
  writing = False
  with open(template_file, "r") as t:
    with open(output_file, "w") as o:
      for line in t:
        if ("IMPORTS GO HERE" in line):
          writing = True
          inject_lines(imports, o)
        elif ("TYPES GO HERE" in line):
          inject_lines(types, o)
        elif ("CLASSES GO HERE" in line):
          inject_lines(class_code, o)
        elif ("PIPELINE GOES HERE"in line):
          inject_with_indent(pipeline, o, 4)
        elif (writing):
            o.write(line)

def extract_types(src, packages):
  with open(src, "r") as f:
    lines = [line.rstrip('\n') for line in f]
    types = []
    for line in lines:
      packages, found = extract_packages(line, packages)
      if not found:
        types.append(line)
  return packages, types

def extract_packages(line, packages):
  found = False
  if "import " in line:
    tree = ast.parse(line)
    for node in ast.walk(tree):      
      if isinstance(node, ast.Import):
        found = True
        for alias in node.names:
          if alias.asname:
            import_statement = f"import {alias.name} as {alias.asname}"
          else:
            import_statement = f"import {alias.name}"
          if import_statement not in packages:
            packages.append(import_statement)
      elif isinstance(node, ast.ImportFrom):
        found = True
        module = node.module
        for alias in node.names:
          if alias.asname:
            import_statement = f"from {module} import {alias.name} as {alias.asname}"
          else:
            import_statement = f"from {module} import {alias.name}"
          if import_statement not in packages:
            packages.append(import_statement)
  return packages, found

def extract_classes(src, packages):
  class_code = []
  for filename in os.listdir(src):
    readline = False
    file_path = os.path.join(src, filename)
    if os.path.isfile(file_path):
      with open(file_path, "r") as f:
        #Read a file line by line
        lines = [line.rstrip('\n') for line in f]
        for line in lines:
          # Extract the import statements
          packages, _ = extract_packages(line, packages)   
          # Extract the class definitions 
          if "class " in line:
            readline = True
          if readline:
            class_code.append(line)
  return packages, class_code

def extract_pipeline(src, process):
  pipeline_lines = []
  pipeline_path = os.path.join(src, "pipeline.py")
  read=True
  pipeline=False
  with open(pipeline_path, 'r') as p:
    lines = [line.rstrip('\n') for line in p]
    for line in lines:
      if f"*** {process.upper()} START ***" in line:
        pipeline=True
      if f"*** {process.upper()} END ***" in line:
        return pipeline_lines
      if pipeline:
        if "	# Inference" in line:
          read=False
        elif "	# Preprocess" in line or 	"# Postprocess" in line:
          read=True
        if read:
          line = line.replace(f"self.{process}_pre", "preprocessor")
          line = line.replace(f"self.{process}_post", "postprocessor")
          pipeline_lines.append(line)

def buildLayerForEachDirectory(dir, temp_dir, types, packages):
  for subdir in os.listdir(dir):
    subdir_path = os.path.join(dir, subdir)
    if os.path.isdir(subdir_path) and "_" not in subdir:  # Only process directories without "_"
      target_dir = os.path.join(temp_dir, subdir)
      os.mkdir(target_dir)
      print(f"Processing directory: {subdir}")
      pipeline = extract_pipeline("src", subdir)
      packages, class_code = extract_classes(subdir_path, packages=packages)
      compiled = os.path.join(target_dir, config['compiled file'])
      build_from_template(config["template file"], compiled, packages, types, class_code, pipeline)

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

  packages = []
  with open(config["template file"], "r") as f:
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
      packages, _ = extract_packages(line, packages)
  packages, types = extract_types("config/typing_config.py", packages=packages)

  ##For each directory in src
  buildLayerForEachDirectory("src", temp_dir, types, packages)

  if(config["package models"]):
    pack_models(temp_dir)

  # Create or update the final destination directory
  if(not os.path.exists(config["Final project location directory"])):
    os.makedirs(config["Final project location directory"], exist_ok=True)

  # Move the contents of temp_dir into the final project directory
  for item in os.listdir(temp_dir):
    s = os.path.join(temp_dir, item)
    d = os.path.join(config["Final project location directory"], item)
    if os.path.isdir(s):
      shutil.copytree(s, d, dirs_exist_ok=True)
    else:
      shutil.copy2(s, d)

  if (config["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  main()