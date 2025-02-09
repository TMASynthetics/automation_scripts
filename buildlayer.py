import tempfile, shutil, os, ast

Settings = {
  "remove tmp files": True,
  "Final project location directory": "aws_lambda"
}


class LayerBuilder:
  def __init__(self):
    self.config = {
      "remove tmp files": True,
      "template file": "/home/josj/scr/MEPS/automation_scripts/templates/lambda_function",
      "compiled file": "lambda_function.py"
    }

  def inject_lines(self, lines, fileHandle):
    for line in lines:
      fileHandle.write(f"{line}\n")

  def inject_with_indent(self, lines, fileHandle, indent):
    # Write each line with a dynamic level of indentation
    if lines is not None:
      lines = [line.strip(" ") for line in lines]
      lines = [line.strip("\t") for line in lines]
      tabs = ' '*indent*2
      for line in lines:
        fileHandle.write(f"{tabs}{line}\n")

  def build_from_template(self, template_file, output_file, imports, types, class_code, inputs, pipeline):
    writing = False
    with open(template_file, "r") as t:
      with open(output_file, "w") as o:
        for line in t:
          if ("IMPORTS GO HERE" in line):
            writing = True
            self.inject_lines(imports, o)
          elif ("TYPES GO HERE" in line):
            self.inject_lines(types, o)
          elif ("CLASSES GO HERE" in line):
            self.inject_lines(class_code, o)
          elif ("LAMBDA INPUTS GO HERE" in line):
            self.inject_with_indent(inputs, o, 4)
          elif ("PIPELINE GOES HERE"in line):
            self.inject_with_indent(pipeline, o, 4)
          elif (writing):
              o.write(line)

  def extract_types(self, src, packages):
    with open(src, "r") as f:
      lines = [line.rstrip('\n') for line in f]
      types = []
      for line in lines:
        packages, found = self.extract_packages(line, packages)
        if not found:
          types.append(line)
    return packages, types

  def extract_packages(self, line, packages):
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

  def extract_classes(self, src, packages):
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
            packages, _ = self.extract_packages(line, packages)   
            # Extract the class definitions 
            if "class " in line:
              readline = True
            if readline:
              class_code.append(line)
    return packages, class_code

  def extract_pipeline(self, src, process):
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

  def extract_pipeline_inputs(self, pipeline):
    parameters = []
    function_call_pattern = re.compile(r'\b\w+\((.*?)\)')

    for line in pipeline:
        matches = function_call_pattern.findall(line)
        for match in matches:
            args = match.split(',')
            for arg in args:
                arg = arg.strip()
                if (
                      not arg.isdigit()
                      and not (arg.startswith('"')
                      and arg.endswith('"'))
                      and not (arg.startswith("'")
                      and arg.endswith("'"))
                      and not arg.startswith('(')
                      and not arg in ['True', 'False']
                      and len(arg) > 0
                    ):
                    parameters.append(f"{arg} = event.get('{arg}')")
    return parameters

  def buildLayerForEachDirectory(self, dir, temp_dir):
    packages = []
    template_packages, types = self.prepare_template(packages)
    for subdir in os.listdir(dir):
      packages = template_packages.copy()
      subdir_path = os.path.join(dir, subdir)
      if os.path.isdir(subdir_path) and "_" not in subdir:  # Only process directories without "_"
        target_dir = os.path.join(temp_dir, subdir)
        os.mkdir(target_dir)
        print(f"Processing directory: {subdir}")
        packages, class_code = self.extract_classes(subdir_path, packages=packages)
        pipeline = self.extract_pipeline("src", subdir)
        inputs = self.extract_pipeline_inputs(pipeline)
        compiled = os.path.join(target_dir, self.config['compiled file'])
        self.build_from_template(self.config["template file"], compiled, packages, types, class_code, inputs, pipeline)

  def buildLayer(self, dir, temp_dir):
    packages = []
    packages, types = self.prepare_template(packages)
    packages, class_code = self.extract_classes(dir, packages=packages)
    pipeline = self.extract_pipeline("src", dir)
    inputs = self.extract_pipeline_inputs(pipeline)
    compiled = os.path.join(temp_dir, self.config['compiled file'])
    self.build_from_template(self.config["template file"], compiled, packages, types, class_code, inputs, pipeline)

  def prepare_template(self, packages):
    with open(self.config["template file"], "r") as f:
      lines = [line.rstrip('\n') for line in f]
      for line in lines:
        packages, _ = self.extract_packages(line, packages)
    packages, types = self.extract_types("config/typing_config.py", packages=packages)
    return packages, types

def run():
  temp_dir = tempfile.mkdtemp()  # Create the temp directory
  print(f"Temporary directory created at: {temp_dir}")

  layerbuilder = LayerBuilder()
  ##For each directory in src
  layerbuilder.buildLayerForEachDirectory("src", temp_dir)

  ## If the processes aren't separated into directories
  ## Uncomment the line below and comment the one above
  # layer.buildLayer("src", packages, temp_dir)

  # Move the contents of temp_dir into the final project directory
  for item in os.listdir(temp_dir):
    s = os.path.join(temp_dir, item)
    d = os.path.join(Settings["Final project location directory"], item)
    if os.path.isdir(s):
      shutil.copytree(s, d, dirs_exist_ok=True)
    else:
      shutil.copy2(s, d)

  if (Settings["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  run()
