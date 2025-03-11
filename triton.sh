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

function run_headless {
  if [ "$(sudo screen -ls | grep "$1")" != "" ]
  then
    echo "❌ $1 is already running"
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
    echo "❌ Couldn't find any projects"
    echo "Please make sure this script is higher up in the directory tree"
    echo "and that this script can reach the project in one of the subdirectories"
    exit 1
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
  dirs=$(find -maxdepth 5 -mindepth 2 -type d -name model_repository | rev | cut -d "/" -f3,2 | rev)
  if [ "$dirs" == "" ]
  then
    echo "Couldn't find any model repositories"
    read -p "Please enter the path to the model repository: " dir
    return $dir
  fi
  select dir in $dirs
  do
    model_repository=$dir
    break
  done
}

function install_tritonconfig.py {
  # Argument $1 is the project directory
  cd "$1"

  wget https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/tritonconfig.py -O tritonconfig.py
  wget https://raw.githubusercontent.com/TMASynthetics/automation_scripts/refs/heads/main/requirements.txt -O .tritonrequirements.txt
  python3 -m venv $tritonconfigenv
  $tritonconfigenv/bin/pip install -r .tritonrequirements.txt
  rm .tritonrequirements.txt
  echo ""
  echo "-- NOTE --"
  echo "We will create a directory in your project to store the Triton Server configuration files"
  echo "You can now choose a name for the model repository"
  read -p "Enter the name for the model repository: " model_repo_name
  $tritonconfigenv/bin/python tritonconfig.py --output $model_repo_name
}

function install {
  if ! check_nvidia_drivers
  then
    echo "Please install NVIDIA drivers first"
    exit 1
  fi

  printf "\n\n== Installing packages! ==\n"
  $package_manager $pkgs
  echo "Done"
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

  # Deeper check
  err=false
  while read model
  do
    model=$model_repository/$model
    if [ "$(ls $model | grep config.pbtxt)" == "" ]
    then
      echo "Error could not find config.pbtxt in $model"
      echo "Try to run tritonconfig.py again and check the path argument"
      echo "Using: $model_repository as model_repository failed"
      err=true
    fi
  done < <(ls $model_repository)
  if $err
  then
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
    --gpus all \
    -p8000:8000 \
    -v $model_repository:/models \
    $triton_image \
    tritonserver \
    --model-repository=/models \
    2>&1 | less
}

function menu {
  PS3="Please select an option: "
  options=("Install/Verify" "Run" "Kill Triton" "Build Triton config files" "Quit")
  select opt in "${options[@]}"
  do
    case $opt in
    "${options[0]}")
      # install
      if ! install
      then
        echo "❌ Installation failed"
        exit 1
      else
        echo "🎉 Installation successful!"
      fi
      ;;
    "${options[1]}")
      # run
      if check_triton_running >/dev/null
      then
        echo "✔ Triton Server is already running"
        break
      fi
      find_model_repository
      echo $model_repository
      run_headless triton_server "sudo $0 run $model_repository"
      sleep 5
      if ! check_triton_running
      then
        echo "run sudo screen -x triton_server to view the output"
        exit 1
      else"$1"/
        echo "🎉 Triton Server started successfully!"
      fi
      ;;
    "${options[2]}")
      # restart
      sudo screen -x triton_server -X quit
      sudo docker stop $(sudo docker ps -q --filter ancestor=$triton_image)
      echo "Triton Server killed"
      ;;
    "${options[3]}")
      # triton config
      find_project_path
      install_tritonconfig.py $project_path
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
    menu
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
  install
elif [ "$1" == "run" ]
then
  run $2
else
  menu
fi