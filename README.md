hearbeat_categorization
==============================

Heartbeat classification based on ECG data, to detect arrythmias and myocardial infarction

run 'pip install -r requirements.txt' to install the environment for this project

1) Model architecture can be found in 'models/model.py'
2) Data, both raw and processed can be accessed using the 'data/' folder
3) Exploratory Data Analysis can be found in the following jupyter notebook: 'notebooks/exploratory/eda.ipynb'
4) A final report of the project is found under the 'reports/' folder, along with figures related to wandb sweep of hyperparameters


# How to run model
1) Clone repo
2) Install MITBIH dataset and store the raw data under 'data/raw/'
3) Added two additional folders under the data directory: 'data/interim/' and 'data/processed/' to store processed datasets that wil be created in the next step
4) run 'src/data/make_dataset.py' to process raw data. Make sure raw data is under correct folder
5) run 'src/models/train_model.py' to start training an validation loop. Note: Logs are recorded through weights and biases and will require a wandb api key/login. This file also has several flags related to hyperparameters, that can be passed along during execution. The default run is 50 epochs with a 0.001 lr
6) Once training run is complete, ensure checkpoints are saved under 'src/models/ckpts/'
7) Run 'src/models/predict_model.py --checkpoint_file "pathToFile"' to run the model on the test dataset on a saved checkpoint. I have added 5 checkpointed models from one of my training runs. This execution prints out model accuracy, a classification report, and generates a confusion matrix, based off predictions on the test dataset.
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
