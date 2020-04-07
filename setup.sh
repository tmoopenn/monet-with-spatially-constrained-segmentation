# setup pyenv
pyenv install 3.7-dev
pyenv global 3.7-dev

# configure system and python dependencies 
sudo apt update
sudo apt-get -y install python3-pip python-pip  build-essential checkinstall software-properties-common
sudo apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# install developer python package
#sudo apt-get -y install python3-dev
#py_version=$(python3 --version)
#echo "Installed python version $py_version"

# alias python and add to bashrc file
#echo 'alias python=python3' >> ~/.bashrc
#source ~/.bashrc

# upgrade pip and install wheel
pip install -U pip
pip install wheel

# Baselines for Atari preprocessing
# Tensorflow is a dependency, but you don't need to install the GPU version
pip install tensorflow
pip install git+git://github.com/openai/baselines

# pytorch-a2c-ppo-acktr for RL utils
pip install git+git://github.com/ankeshanand/pytorch-a2c-ppo-acktr-gail

# Clone and install package
pip install git+git://github.com/mila-iqia/atari-representation-learning.git

# install requirements file 
pip install -r requirements.txt

echo "Python3 has been aliased to python so you can now run python3 by running python"
echo "You can undo this change by removing the alias from your ~/.bashrc file"

