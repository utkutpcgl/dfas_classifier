This model this model was currently trained on `/home/utku/Documents/raw_datasets/datasets_classification/Traffic_sign_light_classification/light/light_dataset_v1_other_decimated_classification_dataset_combined`.


The dataset is generated from multiple datasets combined with the scripts in The dataset is generated from multiple datasets combined with the scripts in `utkusdatamanager/traffic_sign_light/traffic_light`. The code execution order is specified in the code file names and the `README.md` file. 

Filtering small samples seems to improve validation performance metrics, but the model performance is better without filtering small samples as it faces them in the real world (deployment env).

Creating the other dataset, adjusting imbalance parameters, applying active learning (adding errors to train) was most important steps.

The training code is availble at `dfas_classifier/` gitlab repository.
