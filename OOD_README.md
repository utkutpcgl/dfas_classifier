# OOD results

Aşağıda Kuartis Traffic Light Dataset'te (green, red) ResNet50 ile denediğim Out of Distribution (OOD) Detection sonuçlarını paylaşıyorum.

1. Straight softmax OOD

*Açıklama:* 2 yerine 3 sınıf ile (other sınıfı eklenerek) classification yapılıyor. En kolay yöntem bu.

* Sonuçlar:
- OOD (outlier or inlier):
    - accuracy: 0.694
    - f1: 0.82
    - prec: 0.7
    - rec: 0.99
- All classes (red, green or outlier):
    - accuracy: 0.938
    - precision: 0.898
    - recall: 0.948
    - f1: 0.914

* Pros: 
    - Çok basit ve yeterince başırılı. Tune edilecek bir parametre yok.
* Cons: 
    - Recall ve prec ayarlanamıyor, tune edilecek parametere yok.


2. Energy based OOD (https://arxiv.org/abs/2010.03759)

*Açıklama:* Bir resim ile beslenen model (ResNet) çıktıları logitlerden en büyük olanı doğru sınıf kabul ediliyor. Softmax formülünde payda logitlerin exponential toplamını ifade ediyor ve bir örnek in-distribution ise bu değer daha büyük, out-of-dist ise daha küçük çıkıyor. Bu paydanın negatif versyonuna enerji dersek, enerjisi büyük olan resimler out-of-dist sayılıyor. Bu yönteme göre OOD sınıflandırma yapılıyor. Bir model hiç eğitilmeden bunu yapmaya yetkin oluyor, fakat eğitilince (fine-tune) OOD başarısı artıyor. Bu yöntemin normal sınıf başına çıkan softmax çıktılarından daha başarılı bir OOD tespiti yöntemi olduğunu gözlemlemişler. Eğitmek için elinizde kapsayıcı bir OOD datasetinin olması ve m_in ve m_out adındaki hyperparametreleri (hinge loss olarak kullanılıyor) düzgün seçmeniz gerekiyor. Ondan sonra neye in neye out distribution diyeceğinizi bir threshold ile seçmeniz gerekiyor.

* Sonuçlar:
Ben fine-tune ettiğim ResNet50 modelini (öncesinde green ve red traffic light verisinde eğitilmiş) OOD görevindeki başarısını ölçtüm.
- Accuracy at energy_threshold = (m_in + m_out)/2 = 73.37413925019128
- In dist. accuracy 85.8

* Pros: 
    - Recall ve prec ayarlanabiliyor (energy treshold ile).
    - Hali hazırda train edilmiş bir model varsa az epoch ile fine tune edebiliyorsunuz.
* Cons: 
    - Enerji threshold, m_in, m_out seçmek bir dert ve her veri seti için baştan seçilmesi gerekiyor. Yazarlar performansın bunlardan çok etilenmediğini söylese de ben öyle gözlemlemedim.
    - OOD sonuçları hafif daha iyi olsa da in-dist classification sonuçları düşük ve attığımız taş ürküttüğümüz kuşa değmez.
    - OOD ve in-dist detection loss için weightleri seçmek zor (fine tuning için)



3. Sigmoid head for OOD
*Açıklama:* In-distrbiution ve out-distribution iki veri setimiz var ve modelimize bir epochta hem in-distribution verilerinde classification yapmayı (green-red) hem de out of distribution detection yapmayı öğretiyoruz multi-task şeklinde. Burda classification ve OOD detection loss'ları ayrı oluyor ve weightleri adjust etmemiz gerekiyor. 

* In distribution sonuçları iyi olsa da OOD sonuçları bahsetmeye bile değmez şekilde kötü geldi. In distribution ve out-distribution taskleri (lossları) için weight belirlemek zor oldu. Seçebildiğim optimal sonuçta da OOD sonuçları kötü geldi. Sırf OOD taski için eğittiğimde bile (classification taskini bırakıp) sonuçlar çok iyi gelmedi.

* Pros: 
    - Energy based yönteme kıyasla daha kolay parametre tercihi yapılıyor. Sadece task loss weight seçmek gerekiyor. (Burda random weight multi-task learning makalesini denedim). Benim gözlemim makalenin önerisi ile çelişti (rastgele weight vermektense önemli görevin weightini arttırmak lazım.). Denediğim makale: https://openreview.net/forum?id=OdnNBNIdFul .
* Cons: 
    - Enerji threshold, m_in, m_out seçmek bir dert ve her veri seti için baştan seçilmesi gerekiyor. Yazarlar performansın bunlardan çok etilenmediğini söylese de ben öyle gözlemlemedim.
    - OOD sonuçları hafif daha iyi olsa da in-dist classification sonuçları düşük ve attığımız taş ürküttüğümüz kuşa değmez.
