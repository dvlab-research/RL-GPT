# Training PPO-Self-Imitation on MineDojo Tasks
Implementation of RL algorithms, MineDojo environment wrapper and models. 

## Usage
- We provide all the files required in the following document in this **resources link**: **https://disk.pku.edu.cn/#/link/FEEF660CAB0A6B3F3EB55C99729BA198**

- Install MineDojo environment: the official document is: https://docs.minedojo.org/sections/getting_started/install.html#prerequisites. 

Follow my steps:
	- Create python 3.9 environment in anaconda.
	- Install jdk version 171, otherwise you may see some error with Malmo. The package is in the resource link. After installation, you can see the version via `java -version`.
	- Install dependencies `sudo apt install xvfb xserver-xephyr python-opengl ffmpeg`.
	- pip install minedojo
	- If successfully installed, you can run `MINEDOJO_HEADLESS=1 python validate_install.py`.

- Modify MineDojo package: 
	- Delete the official package `pip uninstall minedojo`.
	- Download our repo https://github.com/PKU-RL/MCEnv. Run `python setup.py install`. 
	- For different tasks, carefully check our fast_reset option.

- Install MineCLIP: `pip install git+https://github.com/MineDojo/MineCLIP`, or use the package in the resource link.

- Use PyTorch>=1.8.1. Require x-transformers==0.27.1, otherwise the CLIP model cannot be loaded.

- Check the arguments in train.py.  Download the pretrained MineCLIP model `adjust.pth`  in the resource link.

#### PPO Training

- For PPO, run `MINEDOJO_HEADLESS=1 python train.py`.   
	\-\-task: the programmatic task name.

	\-\-exp-name:  specify dir name prefix of saved logs and models.

	\-\-save-path: this log dir will save models and gifs. 

	Model, gif videos and experience are saved in checkpoint/. Training configs and logs are saved in data/.

- **Draw training curves:** find the training log file progress.txt in data/, move `vis.py` into its directory and run.


## Results

- For milk & wool, the --task is harvest_milk_with_empty_bucket_and_cow and harvest_wool_with_shears_and_sheep. `fig/` shows our training results.

- For other tasks, you may refer to the paper and modify the environment `minecraft.py`, to specify the simulation and reward function.
