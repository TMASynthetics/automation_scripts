import tempfile, shutil, os, ast

config = {
  "remove tmp files": False,
  "template file": "/path/to/automation_scripts/templates/lambda_function",
  "compiled file": "lambda_function.py"
}

def inject_lines(packages, fileHandle):
  for line in packages:
    fileHandle.write(f"{line}\n")

def inject_classes(class_code, fileHandle, indent):
  # Write each line with a dynamic level of indentation
  tabs = '\t'*indent
  for line in class_code:
    fileHandle.write(f"{tabs}{line}\n")

def build_from_template(template_file, output_file, imports, types, class_code):
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
          inject_classes(class_code, o, 0)
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

def extract_pipeline(src, packages):
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

def main():
  temp_dir = tempfile.mkdtemp()  # Create the temp directory
  print(f"Temporary directory created at: {temp_dir}")

  packages = []
  with open(config["template file"], "r") as f:
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
      packages, _ = extract_packages(line, packages)
  packages, types = extract_types("config/typing_config.py", packages=packages)
  packages, class_code = extract_pipeline("src", packages=packages)
  compiled = os.path.join(temp_dir, config["compiled file"])
  build_from_template(config["template file"], compiled, packages, types, class_code)

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
