# Makine Öğrenmesi Modellerinde İstatistiksel Kararlılık Göstergelerine Dayalı Model Yaşlanması Tespiti

## 📋 Proje Hakkında

Bu proje, üretim ortamında çalışan makine öğrenmesi modellerinin zaman içinde yaşadığı performans bozulmalarını (model aging/degradation), **model performansı düşmeden önce** tespit etmeyi amaçlayan yenilikçi bir erken uyarı sistemidir. Geleneksel performans metriklerinin aksine, tahmin hatalarının **istatistiksel kararlılık özellikleri** analiz edilerek concept drift ve veri dağılım değişimleri proaktif olarak yakalanır.

### 🎯 Temel Amaç

Makine öğrenmesi modellerinin üretim ortamındaki en büyük sorunlarından biri "sessiz bozulma"dır (silent degradation). Model, kaynak veri dağılımı değiştikçe yavaş yavaş hatalı tahminler üretmeye başlar, ancak bu bozulma fark edilmeden sistemde kalır. Bu proje:

- **Reaktif değil, proaktif yaklaşım:** Model performansı düşmeden önce degradasyon sinyallerini yakalar
- **İstatistiksel kararlılık temelli:** Residual (tahmin hataları) serisindeki yapısal değişimleri izler
- **Concept drift tespiti:** Veri dağılımındaki kaymayı erken aşamada belirler
- **Otomatik retraining önerisi:** Model yaşlanma eşiğine ulaştığında sistem yenileme sinyali üretir
- **Model agnostik:** Herhangi bir ML/DL modeline uygulanabilir (regression, classification, vs.)

## 🔬 Bilimsel Özgünlük

### Neden Bu Yaklaşım Farklı?

**Geleneksel Yaklaşımlar:**
- Model performansını sadece accuracy, RMSE gibi metriklerle izler
- Performans düştükten *sonra* harekete geçer
- Concept drift tespiti için genellikle referans veri seti gerektirir

**Bu Projenin Yaklaşımı:**
- Tahmin hatalarının **istatistiksel yapısını** sürekli analiz eder
- Hata serisinin **düzensizleşmesi**, **otokorelasyon artışı**, **entropi değişimi** gibi yapısal bozulmaları kullanır
- Ground truth olmadan bile model yaşlanmasını tespit edebilir
- **Model Degradation Index (MDI)** adlı özgün bir metrik geliştirir

Bu yaklaşım, MLOps ve model monitoring literatüründe henüz yeni tartışılmaya başlanan bir araştırma alanıdır ve bu proje akademik yayın potansiyeline sahiptir.

## 🛠️ Teknoloji Stack'i

### Core Libraries
- **Python 3.9+**
- **NumPy & Pandas:** Veri işleme
- **SciPy & Statsmodels:** İstatistiksel testler
  - `scipy.stats`: KPSS, ADF, Levene, KS test
  - `statsmodels.tsa`: Ljung-Box, ACF/PACF
- **Scikit-learn:** ML model eğitimi ve değerlendirme

### ML/DL
- **PyTorch:** Modelleme

### Monitoring & Visualization
- **Matplotlib & Seaborn:** İstatistiksel grafikler
- **Plotly:** İnteraktif MDI dashboard
- **Streamlit / Dash:** Real-time monitoring interface

### Özet Bulgular

1. **Erken Tespit Avantajı** 
2. **Yanlış Alarm Oranı** 
3. **Retraining Verimliliği** 
4. **Model Agnostic**

## 🎓 Akademik Katkılar ve Yayın Potansiyeli

### Özgün Bilimsel Katkılar

1. **Yeni Metrik: Model Degradation Index (MDI)**
   - İstatistiksel kararlılık göstergelerinin birleşimi
   - Concept drift'in erken tespiti için özgün yaklaşım

2. **Proaktif MLOps Paradigması**
   - Performans düşüşü yerine yapısal bozulmayı izleme
   - Ground truth gereksinimi olmayan monitoring

3. **İstatistiksel Temel**
   - Residual stationarity'nin model sağlığı göstergesi olarak kullanımı
   - Entropi ve otokorelasyon bazlı erken uyarı sistemi