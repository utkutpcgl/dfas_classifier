# dfas_classifier

Image classifier that are usually applied after dfas object detection.

Currently in the arac yonelim dataset each tasks first samples are in the train set while the trailer samples are in the validation and test sets. 
I tried to do the same for atis yonelim. However, the sorted order in atis yonelim dataset is corrupted (since 101 is smaller than 9 for alphabetical python sorting.). I have to pad zeros and re-train the atis yonelim dataset and add fix the end-to-end dataset generation code accordingly.

# Plans to improve the classifier:

- *Choosing the mean and std from imagenet is actually common practice. But calculating mean and variance might give better perfromance.*
- Train for some epochs, in the final epochs (with low lr) reload the best weights and fine tune the best weights with low learning rates.
- The model overfits too easily, hence, it should be validated inside of epochs (maybe every half epoch).
- Add focal loss or another type of loss which takes the per sample imbalance into account.

### Using different lr methods:
Sadece AdamW optimizer ile en iyi sonuçları elde ettim iyi bir learning rate seçtikten sonra ve aşağıda SOTA olarak görülen scheduler'lar ile daha kötü sonuçlar elde ettim (ayırca vakit kaybettim):

Cosine annealing: Learning rate yüksek başlayıp kosinüs ile tanımlı bir şeklide azalıyor. Period vesaire seçiliyor.
Ranger21: a synergistic deep learning optimizer. (bir sürü optimizationdan esinlenip daha iyi bir tane elde etmeye çalışıyorlar.).

Özellikle Ranger21'ın daha kötü çalışması İlginç doğrusu. En mantıklı gördüğüm birkaç ayarda train attım, ama sonuçlar yine de daha kötü.

Bunu sebebinin verisetimin yeterince büyük ve representative olmamasından kaynaklı modeliln fazla train olduğunda kolayca overfit olmasından kaynaklı olabileceğini de düşünüyorum. Transfer learning için özel bir optimizer daha faydalı olabilirdi belki benim açımdan.

Yine de, sözde SOTA optimization ve lr scheduling methodları hem daha zor tune ediliyor hem de daha kötü sonuç verebiliyor.


## OOD Results
*Dataset*: Traffic light classification dataset.

*Model*: EfficientNet B0

*AdamW* optimizer withtout lr scheduler:
- best f1: 0.931 (at epoch 9)
- learning rate: 0.0002

*Ranger21* optimizer (repo: https://github.com/lessw2020/Ranger21) (Adam based but with many intrinsic learning rate scheduling methods, i.e. warm-up, warm-down, lookahead etc.):
- best f1: 0.921 (at epoch 7) (with total epochs=10, lr=0.0002, all other settings left default.)
* To many hyperparameters to tune. Changed both warm-up and warm-down (disable/enable), learning rate, epoch numbers.
*Note:* Results depend on the total number of epochs selected as the warm-up and down iteration starting points are determined accordingly. Too many hyperparameters to fine-tune to achieve optimal results.

*AdamW + cosine* annealing lr scheduler (SGDR: STOCHASTIC GRADIENT DESCENT WITHWARM RESTARTS: https://arxiv.org/pdf/1608.03983.pdf):
- best f1: 0.916 (with lr=0.0002 at epoch 2)
* Optimal epoch sayısı ve learning rate bulunduğunda scheduler epoch sayısı total epoch sayısına eşit olduğunda en iyi sonuçları almışlar. Asıl özelliği az epoch ile hızlı converge olmak.


## Traffic Light Results
- Efficient net luke melas (git) results are bad. 
- The current best models imbalance hyp gave good/best results overall. 
- Resnet18 and Effnetb0 results are almost the same. (Hence, no need as effnet is slightly slower.)


# Time elapsed for effnetb0 and resnet18 validation epoch:

* resnet18 4.2 seconds
* effnetb0 5.2 seconds

## Traing tank classification.


### CLASS NAMES WITH ORDER
- {'Tank-M48': 0, 'Tank-M60': 1, 'Tank-leopard': 2}
- {'Green': 0, 'Red': 1, 'other': 2}


