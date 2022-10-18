# dfas_classifier

Image classifier that are usually applied after dfas object detection.

Currently in the arac yonelim dataset each tasks first samples are in the train set while the trailer samples are in the validation and test sets. 
I tried to do the same for atis yonelim. However, the sorted order in atis yonelim dataset is corrupted (since 101 is smaller than 9 for alphabetical python sorting.). I have to pad zeros and re-train the atis yonelim dataset and add fix the end-to-end dataset generation code accordingly.