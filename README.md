# Starter code for final project for CS 378 -- dataset artifact 

Credit: Project by Kaj Bostrom, Jifan Chen, and Greg Durrett. Code by Kaj Bostrom and Jifan Chen.

## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

You should first install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

If you're running on a shared machine and don't have the privileges to install Python packages globally,
or if you just don't want to install these packages permanently, take a look at the "Virtual environments"
section further down in the README.

To make sure pip is installing packages for the right Python version, run `pip --version`
and check that the path it reports is for the right Python interpreter.

## Training and evaluating a model
To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

The default code uses HuggingFace trainer, which checkpoints models periodically (every 500 steps by default). This can be used to make your development cycle faster --  For example, you can start a long training run, leave it going for several hours, and evaluate how much of that time was actually needed to get good performance. If it worked well after only two hours, you'll know that for the future. Ideally, you can then do further experimentation more quickly or even start future runs from checkpoints. 

**Descriptions of other important arguments are available in the comments in `run.py`.**

**If you would like to use different pretrained models, modify the `--model` argument. All available models can be found [here](https://huggingface.co/docs/transformers/v4.16.1/en/model_summary) and [here](https://huggingface.co/models)**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets
This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments
Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.
