import tempfile, shutil, os, ast

config = {
  "remove tmp files": False,
  "template file": "/home/josj/scr/MEPS/xavinator/templates/lambda_function",
  "compiled file": "lambda_function.py"
}

def inject_imports(packages, fileHandle):
  for line in packages:
    fileHandle.write(f"{line}\n")

def inject_classes(class_code, fileHandle, indent):
  # Write each line with a dynamic level of indentation
  tabs = '\t'*indent
  for line in class_code:
    fileHandle.write(f"{tabs}{line}\n")

def build_from_template(template_file, output_file, imports, class_code):
  with open(template_file, "r") as t:
    with open(output_file, "w") as o:
      for line in t:
        if ("IMPORTS GO HERE" in line):
          inject_imports(imports, o)
        elif ("CLASSES GO HERE" in line):
          inject_classes(class_code, o, 0)
        else:
          o.write(line)

def extract_pipeline(src):
  packages = []
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
          if "import " in line:
            tree = ast.parse(line)
            for node in ast.walk(tree):      
              if isinstance(node, ast.Import):
                for alias in node.names:
                  if alias.asname:
                    import_statement = f"import {alias.name} as {alias.asname}"
                  else:
                    import_statement = f"import {alias.name}"
                  if import_statement not in packages:
                    packages.append(import_statement)
              elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                  if alias.asname:
                    import_statement = f"from {module} import {alias.name} as {alias.asname}"
                  else:
                    import_statement = f"from {module} import {alias.name}"
                  if import_statement not in packages:
                    packages.append(import_statement)     
          # Extract the class definitions 
          if "class " in line:
            readline = True
          if readline:
            class_code.append(line)

  return packages, class_code

def main():
  temp_dir = tempfile.mkdtemp()  # Create the temp directory
  print(f"Temporary directory created at: {temp_dir}")

  packages, class_code = extract_pipeline("src")
  compiled = os.path.join(temp_dir, config["compiled file"])
  build_from_template(config["template file"], compiled, packages, class_code)

  if (config["remove tmp files"]):
    print("Removing temp dir")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  main()