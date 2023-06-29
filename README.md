# HYBRID WORKFLOW REPO for QTML

## File structure
* `MPI_regular_hyperband.py`: main script for running regular hyperband. 
* `MPI_swift_hyperband.py`: main script for running swift-hyperband. 
* `utils_MPI.py`: common funcions of `MPI_regular_hyperband.py` and `MPI_swift_hyperband.py`.
* `template_methods.py`: template for the model specific methods.
* `<whatever model>_methods.py`: contains the specific `ConfigGenerator` class and needed methods for `<whatever model>`.
* `start_script_<whatever model>_regular.sh`: batch script to run regular hyperband for `<whatever model>`.
* `start_script_<whatever model>_swift.sh`: batch script to run swift hyperband for `<whatever model>`.
* `predictor.py`: classical performance predictor.
* `qpredictor.py`: quantum performance predictor.
* `QSVR_extended.py`: quantum SVR implementation. Needed by `qpredictor.py`.
* `embeddings.py`: file to save and re-use embeddings for the annealer. Needed by `qpredictor.py`.
* `utility.py`: Needed by `QSVR_extended.py`.
* `create_env-dwave.sh`: create the required environment to run the code.

## How to add a new target model: MyModel
1. Create the file `mymodel_methods.py`.
2. Code the class `ConfigGeneratorMyModel` inside the file `mymodel_methods.py` following the declaration in `template_methods.py`. Taking a look at `ConfigGeneratorLSTM` in `lstm_methods.py` may also be useful.
3. Code the method `train_myModel` inside the file `mymodel_methods.py` following the declaration in `template_methods.py`. Taking a look at `train_lstm` in `lstm_methods.py` may also be useful.
4. Code the additional methods that you may need for saving/loading/building the model and loading the data.
5. Edit `MPI_regular_hyperband.py` and  `MPI_swift_hyperband.py` between the comments that state `CHANGE THIS TO ADD MORE MODELS`. More precisely, add a new case in the `if else` structure as follows:
```python
if model_name == 'some_model':
    ...
elif model_name == 'some_other_model':
    ...
elif model_name == 'mymodel':
    from mymodel_methods import ConfigGeneratorMyModel
    ConfigGenerator_class = ConfigGeneratorMyModel
    from mymodel_methods import train_myModel
    train_model_method = train_myModel
else:
    ...
```
6. Write the batch files `start_script_mymodel_regular.sh` and `start_script_mymodel_swift.sh` to run the algorithms for your model. 

## Main files args

* `MPI_regular_hyperband.py` accepts the following flags:
    - `--dir_name`: (default saved_models_regular) name of the folder where models are saved during training. Must be empty before starting the script.
    - `--seed`: (default 0) random seed for generating configurations.
    - `model_name`: (default lstm) name of the target model.
    - `save_models`: (default False) keep the trained models in `dir_name` once the script is finished.
    - `--r`: R parameter from hyperband algorithm, max epoch that any model is trained.
    - `--eta`: (default 2 the less agressive possible value) eta parameter from hyperband algorithm, controls the discarding rate.



* `MPI_swift_hyperband.py` accepts the following flags:
    - `--dir_name`: (default saved_models_swift) name of the folder where models are saved during training. Must be empty before starting the script.
    - `--seed`: (default 0) random seed for generating configurations.
    - `model_name`: (default lstm) name of the target model.
    - `save_models`: (default False) keep the trained models in `dir_name` once the script is finished.
    - `--r`: R parameter from hyperband algorithm, max epoch that any model is trained.
    - `--eta`: (default 2 the less agressive possible value) eta parameter from hyperband algorithm, controls the discarding rate.
    - `--d`: (default 15) minimum training size for the performance predictors.
    - `--k`: (default 0.5) proportion of models to train, taken from fast-hyperband.
    - `--known_curve`: (default 0.5) position of the performance prediction decision points inside each round.
    - `--pred_type`: (default quantum) quantum | classical.
