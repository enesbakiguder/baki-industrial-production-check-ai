Endüstriyel Ürün Karşılaştırma ve Hata Tespit Sistemi 
Bu proje, üretim hatlarındaki ürünlerin referans görüntülerle karşılaştırılarak hataların (kırık, eksik parça, yanlış konumlandırma vb.) otomatik olarak tespit edilmesi amacıyla geliştirilmiş bir Yapay Zeka (AI) prototipidir.


Bu projeye başladığımda yapay zeka ve görüntü işleme konularında herhangi bir teorik bilgi birikimine sahip değildim. Sadece bir fikirle yola çıktım ve kendi çabalarımla öğrenerek bu noktaya kadar getirdim.

Ancak;
Projeyi ileriye taşıyacak mentorluk veya kurumsal desteğe ulaşamadım.Aynı zamanda donanım ve geliştirme maliyetlerini karşılamada yaşadığım zorluklar nedeniyle projeyi şu anki aşamasında bırakmak zorunda kaldım.

Bu çalışma, kısıtlı imkanlarla dahi nelerin başarılabileceğinin somut bir kanıtı olarak burada durmaktadır.

Githubdan aldığım projenin ilk(saf) haline göre yaptığım değşiklikler:

Proje, karmaşık görüntü işleme tekniklerini bir araya getirir:

Nesne Tespiti 
Sistemin ilk adımı, karmaşık bir görüntü içerisinden "hedef ürünü" bulmaktır.

Projede yolov8n.pt gibi hafif ve hızlı bir model kullanılarak, görüntüdeki ürünün koordinatları anlık olarak belirlenir.

Orijinal projeden farklı olarak, YOLO sadece bir kutu çizmekle kalmaz; sistemin geri kalanının (segmentasyon ve eşleştirme) sadece bu odak noktası üzerinde çalışmasını sağlayarak işlem yükünü azaltır ve hata payını düşürür.

Akıllı Segmentasyon - GrabCut Algoritması
Nesne tespitinden sonra, ürünün çevresindeki gereksiz arka plan gürültüsünü temizlemek için bu aşama devreye girer.

YOLO'dan gelen koordinatlar GrabCut algoritmasına girdi olarak verilir. Algoritma, pikselleri "kesin ön plan" ve "kesin arka plan" olarak sınıflandırır.

Bu sayede ürünün gölgesi, üzerinde durduğu bant veya arkadaki karmaşık nesneler eşleştirme sürecine dahil edilmez. Sadece "saf ürün" verisi bir sonraki aşamaya aktarılır.

Gelişmiş Görüntü Eşleştirme 
Bu aşama projenin "beyni"dir ve iki görüntü arasındaki benzerliği matematiksel olarak kanıtlar.

Renk Bağımsızlığı (Norm Yapısı): Tarafından eklenen normalizasyon katmanı ile görüntüler siyah-beyaz hale getirilir ve CLAHE ile ışık dengelenir. Bu, sistemin fabrikanın ışığından veya ürünün renginden bağımsız olarak sadece geometrik dokuya odaklanmasını sağlar.

LightGlue Teknolojisi: Geleneksel eşleştiricilerin aksine, LightGlue derin öğrenme tabanlıdır ve hangi noktaların "güvenilir" olduğunu anlar. SIFT ile bulunan anahtar noktalar arasında en tutarlı eşleşmeleri (inliers) belirleyerek benzerlik skorunu üretir.

Değişim Haritası
Sadece "farklı" demek yerine, "neresi farklı" sorusuna görsel cevap verilir.

İki görüntü üst üste bindirildikten sonra aralarındaki farklar hesaplanır.

Referans görüntü ile sorgu görüntüsü arasındaki sapmalar kırmızı piksellerle işaretlenir. Eğer bu kırmızı alanların toplam ürün alanına oranı belirlenen eşiği geçerse, sistem otomatik olarak "GEÇERSİZ" kararı verir.

Neden Bu Yapı Çok Önemli:
Bu proje sadece bir eşleştirme kodu değil; YOLO (tespit), GrabCut (hassas kesim), Normalizasyon (ışık direnci) ve LightGlue (modern AI eşleme) tekniklerini birleştiren hibrit bir kalite kontrol istasyonu prototipidir.


Dosya Yapısı
matching/: Eşleştirme algoritmalarının çekirdek dosyaları.

product_change/: Kod tarafından üretilen görsel analiz raporları.

yolov8n.pt: Kullanılan eğitilmiş model dosyası.

temp.py : Projenin ana çalışma dosyası.

Kullanım ve Lisans
Bu proje tamamen ücretsiz ve açık kaynaklıdır ancak minik bir özel ricam bulunmaktadır:

Bu projenin kodlarını veya mantığını herhangi bir yerde kullanmadan önce lütfen sadece bana danışın. Başka hiçbir beklentim veya kısıtlamam yoktur.

Nasıl Çalıştırılır?
Gerekli kütüphaneleri yükleyin: pip install ultralytics opencv-python numpy.

Referans resimlerinizi reference_images klasörüne atın.

demo.py dosyasındaki QUERY_IMAGE_PATH kısmını test etmek istediğiniz resimle güncelleyip çalıştırın.






NOT:PROJENİN İLK HALİNİN GELİŞTİRME AŞAMASINA BAKMAK İÇİN:
