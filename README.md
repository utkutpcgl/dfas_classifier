# dfas_classifier

Image classifier that are usually applied after dfas object detection.

Currently in the arac yonelim dataset each tasks first samples are in the train set while the trailer samples are in the validation and test sets. 
I tried to do the same for atis yonelim. However, the sorted order in atis yonelim dataset is corrupted (since 101 is smaller than 9 for alphabetical python sorting.). I have to pad zeros and re-train the atis yonelim dataset and add fix the end-to-end dataset generation code accordingly.

# Plans to improve the classifier:

- Use different learning rate schedulers. 
- Add focal loss or another type of loss which takes the per sample imbalance into account.

# Time elapsed for effnetb0 and resnet18 validation epoch:

resnet18 4.2 seconds

effnetb0 5.2 seconds

## Traing tank classification.


### CLASS NAMES WITH ORDER
{'Tank-M48': 0, 'Tank-M60': 1, 'Tank-leopard': 2}