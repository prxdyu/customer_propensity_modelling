# printing the date followed by START
echo [$(date)]: "START"

# printing the data followed by string "creating env"
echo [$(date)] : "Creating env with python 3.8 version "

# creating the environment in the env folder automatically
conda create --prefix ./env python=3.8 -y

echo [$(date)] : "Activating the environment "

# activating the environment
conda activate ./env

# installing the dev requirements
echo [$(date)] : "Installing the requirements"
pip install -r requirements.txt

echo [$(date)] : "END"