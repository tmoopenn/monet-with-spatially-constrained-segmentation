
# configure system and python dependencies 
sudo apt update
sudo apt-get -y install python3-pip  build-essential checkinstall software-properties-common
sudo apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

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
pip install git+git://github.com/tmoopenn/atari-representation-learning.git

# install requirements file 
pip install -r requirements.txt


