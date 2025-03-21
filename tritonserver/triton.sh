#!/bin/bash

triton_image=nvcr.io/nvidia/tritonserver:25.02-py3
pkgs="docker.io nvidia-container-toolkit screen"
package_manager="sudo apt install -y"
tritonconfigenv=".tritonconfigenv"
model_repository=""
project_path=""
wd=$(pwd)

function check_nvidia_drivers {
  if lspci | grep -i nvidia &> /dev/null
  then
    echo "✔ NVIDIA GPU found"
    if ! nvidia-smi &> /dev/null
    then
      echo "❌ NVIDIA drivers not found"
      return 1
    else
      echo "✔ NVIDIA drivers already installed"
    fi
  else
    echo "❌ NVIDIA GPU not found"
    return 1
  fi
  return 0
}

function check_docker {
  if ! systemctl is-active --quiet docker
  then
    echo "❌ Docker failed to start"
    return 1
  fi
}

function check_triton_image {
  if ! sudo docker image inspect $triton_image &> /dev/null
  then
    echo "❌ Triton Server image not found in local registry"
    return 1
  fi
}

function check_nvidia_container_runtime {
  if ! grep -q '"nvidia":' /etc/docker/daemon.json 2>/dev/null
  then
    echo "❌ NVIDIA runtime is not configured"
    return 1
  fi
}

function check_triton_running {
  triton="$(sudo docker ps | grep $triton_image)"
  if [ "$triton" == "" ]
  then
    echo "❌ Triton Server is not running"
    return 1
  fi
}

function check_model_repository_files {
  if ! check_model_repository_path $model_repository
  then 
    return 1
  fi

  err=false
  while read model
  do
    model=$model_repository/$model
    if [ "$(ls $model | grep config.pbtxt)" == "" ]
    then
      err=true
    fi
  done < <(ls $model_repository)
  if $err
  then
    echo "❌ Using: $model_repository as model_repository failed"
    echo "Please check the path"
    echo "Or try to run build the model_repository again"
    return 1
  else
    return 0
  fi
}

function check_model_repository_path {
    # Quick sanity check
  if [ ! -d "$1" ]
  then
    echo "Usage $0 <path to model_repository>"
    return 1
  elif [ -d "$1/model_repository" ]
  then
    model_repository="$1/model_repository"
  else
    model_repository="$1"
  fi
  repo_root=$(echo $model_repository | cut -d "/" -f1)
  real_root=$(pwd | cut -d "/" -f1)
  if [ "$repo_root" != "$real_root" ]
  then
    model_repository=$(pwd)/$model_repository
  fi
}

function run_headless {
  if [ "$(sudo screen -ls | grep "$1")" != "" ]
  then
    echo "❌ screen session $1 is already running"
    echo  "Try to kill triton server first"
    return 1
  else
    echo "Starting $2 in the background"
    echo "Run 'sudo screen -x $1' to view the output"
    sudo screen -dmS "$1" 
    sudo screen -x "$1" -X stuff "$2\n"
  fi
}

function find_project_path {
  echo "Finding projects in the current directory"
  dirs=$(find -maxdepth 5 -mindepth 2 -type f -name models.py | rev | cut -d "/" -f5,4,3 | rev)
  if [ "$dirs" == "" ]
  then
    echo "⁉️ Couldn't find any projects"
    read -p "Do you want to manually enter the path to the project? [y/n]: " choice
    if [ "$choice" == "y" ]
    then
      read -p "Enter the path to the project: " project_path
      return 0
    else
      return 1
    fi
  fi
  echo "Below are the projects found: "
  select dir in $dirs
  do
    project_path=$dir
    break
  done
}

function find_model_repository {
  echo "Finding model repositories in the current directory"
  echo "Below are the model repositories found"
  dirs=$(find -maxdepth 5 -mindepth 2 -type d -name model_repository | rev | cut -d "/" -f5,4,3,2 | rev)
  if [ "$dirs" == "" ]
  then
    echo "⁉️ Couldn't find any model repositories"
    read -p "Please enter the path to the model repository: " model_repository
    return 0
  fi
  select dir in $dirs
  do
    model_repository=$dir
    break
  done
}

function build_tritonconfig.py {
  # Argument $1 is the project directory
  cd "$1"

  wget https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/tritonconfig.py -O tritonconfig.py
  wget https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/requirements.txt -O .tritonrequirements.txt
  python3 -m venv $tritonconfigenv
  $tritonconfigenv/bin/pip install -r .tritonrequirements.txt
  rm .tritonrequirements.txt

}

function build_model_repository {
  if ! find_project_path
  then
    read -p "Do you want to manually enter the path to the models directory? [y/n]: " choice
    if [ "$choice" == "y" ]
    then
      read -p "Enter the path to the models directory: " input_models
      read -p "Enter the ouptput directory for model_repository: " model_repo_name
      build_tritonconfig.py "/tmp"
      $tritonconfigenv/bin/python tritonconfig.py --models "$input_models" --output "$model_repo_name"
    else
      echo "❌ Couldn't find any projects"
      echo "[TIP] Copy this script either to your project directory or a parent of the project"
      echo "Then run the script again"
      return 1
    fi
  else
    build_tritonconfig.py "$project_path"
    echo ""
    echo "-- NOTE --"
    echo "We will create a directory in your project to store the Triton Server configuration files"
    echo "You can now choose a name for the model repository"
    read -p "Enter the name for the model repository: " model_repo_name
    pwd
    $tritonconfigenv/bin/python tritonconfig.py --output $model_repo_name
    rm -rf $tritonconfigenv
    echo "Triton Server configuration files created in $1/$model_repo_name"
    echo "You can now select 'Run' to start Triton Server"
  fi
}

function install_triton {
  if ! check_nvidia_drivers
  then
    echo "Please install NVIDIA dricdfvers first"
    echo "🔗 https://www.nvidia.com/drivers/"
    exit 1
  fi

  printf "\n\n== Installing packages! ==\n"
  $package_manager $pkgs
  echo "Done"
  echo ""
  for i in {5..1}
  do
    printf "\r== Verifying Docker in $i seconds == "
    sleep 1
  done
  printf "\r== Verifying Docker ==              \n"
  if ! check_docker
  then
    echo "Trying to start it manually..."
    sudo systemctl stop docker
    sudo systemctl start docker
    sudo systemctl enable docker
    if ! check_docker
    then
      exit 1
    fi
  fi
  echo "✔ Docker started successfully"

  printf "\n\n== Verifying Triton Server Image ==\n"
  if ! check_triton_image
  then
    echo "Pulling image: $triton_image"
    echo "Go Grab a coffee ☕... or maybe a holiday 🏖️"
    echo ""
    if sudo docker pull $triton_image
    then
      echo "✔ Triton Server image pulled successfully"
    else
      echo "❌ Failed to pull Triton Server image"
      exit 1
    fi
  else
    echo "✔ Triton Server image already exists, skipping pull."
  fi

  printf "\n\n== Verifying NVIDIA Runtime ==\n"
  if ! check_nvidia_container_runtime
  then
    sudo nvidia-ctk runtime configure --runtime=docker
  else
    echo "✔ NVIDIA runtime already configured."
  fi

  printf "\n\n== Restarting Docker ==\n"
  sudo systemctl restart docker
  if ! check_docker
  then
    echo "❌ Docker failed to start"
    return 1
  fi
  echo "Done"

  return 0
}
                                                                                                                                                                                                                                                                    
function run {
  if ! check_model_repository_path $1
  then
    echo "❌ Model repository not found"
    return 1
  fi

  if ! check_model_repository_files
  then
    echo "❌ The Model repository is invalid"
    echo "Please run the script again and provide a valid model repository"
    return 1
  fi

  if ! check_triton_image
  then
    echo "❌ Triton Server image not found in local registry"
    return 1
  fi
  if ! check_docker
  then
    echo "❌ Docker failed to start"
    return 1
  fi
  if ! check_nvidia_container_runtime
  then
    echo "❌ NVIDIA runtime is not configured"
    return 1
  fi
  echo "pre-flight Checks complete"

  sudo docker run \
    --rm \
    --runtime=nvidia \
    --gpus all \
    -p8000:8000 \
    -v $model_repository:/models \
    $triton_image \
    tritonserver \
    --model-repository=/models \
    2>&1 | less
}

function menu {
  PS3="Menu: "
  options=("Install/Verify" "Build Triton config files" "Run" "Kill Triton" "Quit")
  select opt in "${options[@]}"
  do
    case $opt in
    "${options[0]}")
      # install
      if ! install_triton
      then
        echo "❌ Installation failed"
        exit 1
      else
        echo "🎉 Installation successful!"
      fi
      ;;
    "${options[1]}")
      # model_repository
      if ! build_model_repository
      then
        echo "❌ Failed to build model repository"
        break
      else
        echo "🎉 Model repository created successfully!"
      fi
      ;;
    "${options[2]}")
      # run
      if check_triton_running >/dev/null
      then
        echo "✔ Triton Server is already running"
        break
      fi
      find_model_repository
      if ! check_model_repository_path $model_repository
      then
        echo "❌ Model repository not found"
        break
      fi
      echo "Model_repository: $model_repository"
      if ! check_model_repository_files
      then
        echo "❌ Stopping"
        break
      fi
      if ! run_headless triton_server "sudo $0 run $model_repository"
      then
        echo "❌ Failed to start Triton Server"
        break
      else
        sleep 5
        if ! check_triton_running
        then
          echo "run sudo screen -x triton_server to view the output"
        else
          echo "🎉 Triton Server started successfully!"
        fi
      fi
      ;;
    "${options[3]}")
      # kill
      sudo screen -x triton_server -X quit
      sudo docker stop $(sudo docker ps -q --filter ancestor=$triton_image) 2>/dev/null
      echo "Triton Server killed"
      ;;
    "${options[4]}")
      # quit
      echo "Goodbye!"
      exit 0
      ;;
    *)
      echo "Invalid option $REPLY"
      ;;
    esac
    cd $wd
    echo ""
    i=0
    for option in "${options[@]}"
    do
      i=$((i+1))
      if [ $(($i % 4)) == 0 ]
      then
        echo ""
      fi
      printf "%d) %s  " $i "$option"
    done
    echo ""
    echo "Please select an option: "
  done
}

#Start triton server
echo "============================================================"
echo " _____     _ _              ____                           "
echo "|_   _| __(_) |_ ___  _ __ / ___|  ___ _ ____   _____ _ __ "
echo "  | || '__| | __/ _ \\| '_ \\\\___ \\ / _ \\ '__\\ \\ / / _ \\ '__|"
echo "  | || |  | | || (_) | | | |___) |  __/ |   \\ V /  __/ |   "
echo "  |_||_|  |_|\\__\\___/|_| |_|____/ \\___|_|    \\_/ \\___|_|   "
echo "============================================================"
echo ""
echo "Welcome to Triton Server setup script"
echo "This script will help you install and run Triton Server"
echo "Please select an option from the menu below"
echo ""

if [ "$1" == "install" ]
then
  install_triton
elif [ "$1" == "run" ]
then
  run $2
else
  menu
fi