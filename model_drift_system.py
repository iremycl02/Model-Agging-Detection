"""
Makine Öğrenmesi Model Yaşlanması Tahmin ve Adaptasyon Sistemi
================================================================
Bu sistem, zaman içinde model performansının düşüşünü tespit eder,
tahmin eder ve modeli otomatik olarak adapte eder.

Özellikler:
- Concept Drift Tespiti (ADWIN, Page-Hinkley, DDM)
- Data Drift Analizi (KS Test, PSI, Wasserstein Distance)
- Model Performance Degradation Prediction
- Adaptive Learning (Online Learning, Incremental Learning)
- Multi-Model Support
- Real-time Monitoring Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Deep Learning kütüphaneleri
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Grafik ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("MODEL YAŞLANMASI TAHMİN VE ADAPTASYON SİSTEMİ")
print("=" * 80)

# ============================================================================
# 1. DRIFT DETECTION ALGORİTMALARI
# ============================================================================

class ADWIN:
    """
    Adaptive Windowing (ADWIN) - Concept Drift Detector
    Veri akışındaki değişimleri otomatik olarak tespit eder
    """
    def __init__(self, delta=0.002):
        self.delta = delta
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
    def add_element(self, value):
        """Yeni eleman ekle ve drift kontrol et"""
        self.window.append(value)
        self.width += 1
        
        if self.width == 1:
            self.total = value
            self.variance = 0.0
            return False
            
        self.total += value
        
        # Drift detection
        drift_detected = self._detect_change()
        
        if drift_detected:
            # Pencereyi küçült
            cut_point = self.width // 2
            self.window = self.window[cut_point:]
            self.width = len(self.window)
            self.total = sum(self.window)
            
        return drift_detected
    
    def _detect_change(self):
        """Değişim tespiti"""
        if self.width < 2:
            return False
            
        # İki pencere oluştur ve karşılaştır
        for i in range(1, self.width):
            w0 = self.window[:i]
            w1 = self.window[i:]
            
            n0, n1 = len(w0), len(w1)
            if n0 < 5 or n1 < 5:
                continue
                
            mean0, mean1 = np.mean(w0), np.mean(w1)
            
            # Hoeffding bound
            m = 1.0 / (1.0/n0 + 1.0/n1)
            epsilon = np.sqrt((1.0/(2*m)) * np.log(4.0/self.delta))
            
            if abs(mean0 - mean1) > epsilon:
                return True
                
        return False


class PageHinkley:
    """
    Page-Hinkley Test - Cumulative difference drift detector
    """
    def __init__(self, threshold=50, alpha=0.9999):
        self.threshold = threshold
        self.alpha = alpha
        self.sum = 0
        self.min_sum = 0
        self.mean = 0
        self.n = 0
        
    def add_element(self, value):
        """Yeni değer ekle ve drift kontrol et"""
        self.n += 1
        
        # Running mean
        self.mean = self.mean + (value - self.mean) / self.n
        
        # Cumulative difference
        self.sum = self.alpha * self.sum + (value - self.mean)
        
        # Minimum cumulative sum
        if self.sum < self.min_sum:
            self.min_sum = self.sum
            
        # Drift detection
        drift = (self.sum - self.min_sum) > self.threshold
        
        if drift:
            self.reset()
            
        return drift
    
    def reset(self):
        """Detector'ı sıfırla"""
        self.sum = 0
        self.min_sum = 0
        self.mean = 0


class DDM:
    """
    Drift Detection Method (DDM)
    Hata oranındaki artışları tespit eder
    """
    def __init__(self, warning_level=2.0, drift_level=3.0):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()
        
    def reset(self):
        self.error_rate = 0
        self.std = 0
        self.min_error_rate = float('inf')
        self.min_std = float('inf')
        self.n = 0
        
    def add_element(self, prediction_correct):
        """
        Tahmin sonucu ekle
        prediction_correct: True (doğru) veya False (yanlış)
        """
        self.n += 1
        error = 0 if prediction_correct else 1
        
        # Running statistics
        self.error_rate += (error - self.error_rate) / self.n
        self.std = np.sqrt(self.error_rate * (1 - self.error_rate) / self.n)
        
        # Minimum değerleri güncelle
        if self.error_rate + self.std < self.min_error_rate + self.min_std:
            self.min_error_rate = self.error_rate
            self.min_std = self.std
            
        # Drift detection
        warning = False
        drift = False
        
        if self.n > 30:  # Minimum sample size
            if self.error_rate + self.std > self.min_error_rate + self.warning_level * self.min_std:
                warning = True
                
            if self.error_rate + self.std > self.min_error_rate + self.drift_level * self.min_std:
                drift = True
                self.reset()
                
        return {'warning': warning, 'drift': drift}


# ============================================================================
# 2. DATA DRIFT ANALYSIS
# ============================================================================

class DataDriftAnalyzer:
    """
    Veri dağılımındaki değişimleri analiz eder
    """
    def __init__(self, reference_data):
        """
        Args:
            reference_data: Referans (training) veri seti
        """
        self.reference_data = reference_data
        self.feature_names = reference_data.columns.tolist()
        
    def kolmogorov_smirnov_test(self, current_data, alpha=0.05):
        """
        Kolmogorov-Smirnov testi ile drift tespiti
        """
        results = {}
        
        for feature in self.feature_names:
            if feature in current_data.columns:
                ref_values = self.reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                # KS test
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                
                results[feature] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < alpha,
                    'drift_score': statistic
                }
                
        return results
    
    def population_stability_index(self, current_data, n_bins=10):
        """
        Population Stability Index (PSI) hesaplama
        PSI > 0.2: Significant drift
        PSI > 0.1: Moderate drift
        """
        results = {}
        
        for feature in self.feature_names:
            if feature in current_data.columns:
                ref_values = self.reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                # Binning
                min_val = min(ref_values.min(), curr_values.min())
                max_val = max(ref_values.max(), curr_values.max())
                bins = np.linspace(min_val, max_val, n_bins + 1)
                
                ref_hist, _ = np.histogram(ref_values, bins=bins)
                curr_hist, _ = np.histogram(curr_values, bins=bins)
                
                # Normalize
                ref_prop = (ref_hist + 1) / (len(ref_values) + n_bins)  # Laplace smoothing
                curr_prop = (curr_hist + 1) / (len(curr_values) + n_bins)
                
                # PSI calculation
                psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
                
                results[feature] = {
                    'psi': psi,
                    'drift_level': 'High' if psi > 0.2 else ('Moderate' if psi > 0.1 else 'Low')
                }
                
        return results
    
    def wasserstein_distance_test(self, current_data):
        """
        Wasserstein (Earth Mover's) Distance ile drift ölçümü
        """
        results = {}
        
        for feature in self.feature_names:
            if feature in current_data.columns:
                ref_values = self.reference_data[feature].dropna().values
                curr_values = current_data[feature].dropna().values
                
                distance = wasserstein_distance(ref_values, curr_values)
                
                # Normalize (0-1 arası)
                ref_range = ref_values.max() - ref_values.min()
                normalized_distance = distance / ref_range if ref_range > 0 else 0
                
                results[feature] = {
                    'distance': distance,
                    'normalized_distance': normalized_distance
                }
                
        return results
    
    def comprehensive_drift_report(self, current_data):
        """Kapsamlı drift raporu"""
        print("\n" + "="*80)
        print("DATA DRIFT ANALYSIS REPORT")
        print("="*80)
        
        # KS Test
        print("\n1. Kolmogorov-Smirnov Test Results:")
        print("-" * 80)
        ks_results = self.kolmogorov_smirnov_test(current_data)
        
        drift_features = []
        for feature, result in ks_results.items():
            if result['drift_detected']:
                drift_features.append(feature)
                print(f"   {feature}: DRIFT DETECTED (p={result['p_value']:.4f}, stat={result['statistic']:.4f})")
        
        if not drift_features:
            print("   No significant drift detected in any feature")
            
        # PSI
        print("\n2. Population Stability Index (PSI):")
        print("-" * 80)
        psi_results = self.population_stability_index(current_data)
        
        for feature, result in psi_results.items():
            print(f"   {feature}: PSI={result['psi']:.4f} ({result['drift_level']} drift)")
            
        # Wasserstein Distance
        print("\n3. Wasserstein Distance:")
        print("-" * 80)
        wd_results = self.wasserstein_distance_test(current_data)
        
        for feature, result in wd_results.items():
            print(f"   {feature}: Distance={result['normalized_distance']:.4f}")
            
        return {
            'ks_test': ks_results,
            'psi': psi_results,
            'wasserstein': wd_results,
            'drift_features': drift_features
        }


# ============================================================================
# 3. MODEL PERFORMANCE DEGRADATION PREDICTOR
# ============================================================================

class ModelDegradationPredictor:
    """
    Model performans düşüşünü tahmin eder
    """
    def __init__(self):
        self.performance_history = []
        self.timestamps = []
        self.prediction_model = None
        
    def add_performance_metric(self, timestamp, metrics):
        """
        Performans metriği ekle
        Args:
            timestamp: Zaman damgası
            metrics: Dict of metrics (accuracy, f1, etc.)
        """
        self.timestamps.append(timestamp)
        self.performance_history.append(metrics)
        
    def detect_degradation(self, window_size=10, threshold=0.05):
        """
        Performans düşüşü tespit et
        """
        if len(self.performance_history) < window_size:
            return {'degradation_detected': False}
            
        # Son window_size metrikleri al
        recent_metrics = self.performance_history[-window_size:]
        
        # Her metrik için trend analizi
        results = {}
        for metric_name in recent_metrics[0].keys():
            values = [m[metric_name] for m in recent_metrics]
            
            # Linear regression ile trend
            x = np.arange(len(values)).reshape(-1, 1)
            y = np.array(values)
            
            slope = np.polyfit(x.flatten(), y, 1)[0]
            
            # Düşüş var mı?
            degradation = slope < -threshold
            
            results[metric_name] = {
                'trend_slope': slope,
                'degradation_detected': degradation,
                'current_value': values[-1],
                'change_rate': slope / values[0] if values[0] != 0 else 0
            }
            
        return results
    
    def predict_future_performance(self, steps_ahead=5):
        """
        Gelecekteki performansı tahmin et
        """
        if len(self.performance_history) < 10:
            return None
            
        predictions = {}
        
        for metric_name in self.performance_history[0].keys():
            values = [m[metric_name] for m in self.performance_history]
            
            # Time series forecasting (Simple linear extrapolation)
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, deg=2)  # 2nd degree polynomial
            
            # Predict future
            future_x = np.arange(len(values), len(values) + steps_ahead)
            future_values = np.polyval(coeffs, future_x)
            
            predictions[metric_name] = {
                'predicted_values': future_values.tolist(),
                'trend': 'declining' if coeffs[0] < 0 else 'stable/improving'
            }
            
        return predictions
    
    def visualize_performance_trend(self):
        """Performans trendini görselleştir"""
        if len(self.performance_history) < 2:
            print("Insufficient data for visualization")
            return
            
        metrics = list(self.performance_history[0].keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric_name in enumerate(metrics):
            values = [m[metric_name] for m in self.performance_history]
            
            axes[idx].plot(self.timestamps, values, marker='o', linewidth=2, label='Actual')
            
            # Trend line
            x = np.arange(len(values))
            z = np.polyfit(x, values, 2)
            p = np.poly1d(z)
            axes[idx].plot(self.timestamps, p(x), '--', alpha=0.7, label='Trend')
            
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel(metric_name.upper())
            axes[idx].set_title(f'{metric_name.upper()} Over Time')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()


# ============================================================================
# 4. ADAPTIVE LEARNING SYSTEM
# ============================================================================

class AdaptiveModelSystem:
    """
    Adaptif model sistemi - Modeli otomatik olarak günceller
    """
    def __init__(self, base_model, retrain_threshold=0.05):
        """
        Args:
            base_model: Sklearn veya PyTorch modeli
            retrain_threshold: Yeniden eğitim için performans düşüş eşiği
        """
        self.base_model = base_model
        self.retrain_threshold = retrain_threshold
        self.current_model = None
        self.performance_monitor = ModelDegradationPredictor()
        self.drift_detector = DDM()
        
    def initial_train(self, X_train, y_train):
        """İlk eğitim"""
        print("\n[INFO] Initial model training...")
        self.current_model = self.base_model
        self.current_model.fit(X_train, y_train)
        print("[SUCCESS] Initial training completed")
        
    def predict_and_adapt(self, X, y_true=None, adaptation_mode='incremental'):
        """
        Tahmin yap ve gerekirse adapte ol
        
        Args:
            X: Features
            y_true: True labels (if available)
            adaptation_mode: 'incremental' or 'retrain'
        """
        # Tahmin
        predictions = self.current_model.predict(X)
        
        # Eğer gerçek etiketler varsa, performansı izle
        if y_true is not None:
            # Drift detection
            for pred, true in zip(predictions, y_true):
                is_correct = (pred == true)
                drift_status = self.drift_detector.add_element(is_correct)
                
                if drift_status['drift']:
                    print("\n[WARNING] Concept drift detected!")
                    
                    if adaptation_mode == 'incremental':
                        self._incremental_update(X, y_true)
                    elif adaptation_mode == 'retrain':
                        print("[INFO] Retraining recommended")
                        
        return predictions
    
    def _incremental_update(self, X_new, y_new):
        """
        Incremental learning (Online learning)
        """
        print("[INFO] Performing incremental update...")
        
        # SGDClassifier için partial_fit kullan
        if hasattr(self.current_model, 'partial_fit'):
            self.current_model.partial_fit(X_new, y_new)
            print("[SUCCESS] Model updated incrementally")
        else:
            print("[WARNING] Model doesn't support incremental learning")
            print("[INFO] Consider using SGDClassifier or online learning models")
            
    def evaluate_and_record(self, X_test, y_test, timestamp):
        """Model performansını değerlendir ve kaydet"""
        y_pred = self.current_model.predict(X_test)
        y_pred_proba = self.current_model.predict_proba(X_test)[:, 1] if hasattr(self.current_model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            
        self.performance_monitor.add_performance_metric(timestamp, metrics)
        
        return metrics


# ============================================================================
# 5. PYTORCH ADAPTIVE NEURAL NETWORK
# ============================================================================

class AdaptiveNeuralNetwork(nn.Module):
    """
    PyTorch tabanlı adaptif neural network
    Online learning ve fine-tuning destekler
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2, dropout=0.3):
        super(AdaptiveNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class PyTorchAdaptiveSystem:
    """PyTorch modeller için adaptif sistem"""
    
    def __init__(self, input_dim, learning_rate=0.001, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = AdaptiveNeuralNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        
    def train_initial(self, X_train, y_train, epochs=50, batch_size=256):
        """İlk eğitim"""
        print(f"\n[INFO] Training PyTorch model on {self.device}...")
        
        # Veriyi hazırla
        X_scaled = self.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_train.values).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
                
        print("[SUCCESS] Initial training completed")
        
    def incremental_update(self, X_new, y_new, epochs=5, batch_size=128):
        """Incremental learning"""
        print("[INFO] Performing incremental update...")
        
        X_scaled = self.scaler.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_new.values).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
        print("[SUCCESS] Incremental update completed")
        
    def predict(self, X):
        """Tahmin"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """Olasılık tahmini"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
            
        return probas.cpu().numpy()


# ============================================================================
# 6. MAIN SYSTEM - ENTİTY SİSTEM
# ============================================================================

class ModelLifecycleManager:
    """
    Tüm sistemi yöneten ana sınıf
    """
    def __init__(self, data_path=None, dataframe=None):
        """
        Args:
            data_path: CSV dosya yolu
            dataframe: Pandas DataFrame (alternatif)
        """
        print("\n" + "="*80)
        print("MODEL LIFECYCLE MANAGER INITIALIZATION")
        print("="*80)
        
        # Veriyi yükle
        if dataframe is not None:
            self.df = dataframe
        elif data_path:
            print(f"\n[INFO] Loading data from {data_path}...")
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or dataframe must be provided")
            
        print(f"[SUCCESS] Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Temel bileşenler
        self.drift_detector = None
        self.data_drift_analyzer = None
        self.adaptive_system = None
        self.pytorch_system = None
        
        # Zaman simülasyonu için
        self.current_time = 0
        
    def prepare_data(self, target_column='Class', test_size=0.3):
        """Veriyi hazırla"""
        print("\n[INFO] Preparing data...")
        
        # Feature ve target ayır
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scaling
        self.scaler = RobustScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"[SUCCESS] Data prepared - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
        # Class distribution
        print(f"\nClass distribution:")
        print(f"   Training: {dict(self.y_train.value_counts())}")
        print(f"   Test: {dict(self.y_test.value_counts())}")
        
    def initialize_systems(self, model_type='sklearn'):
        """Sistemleri başlat"""
        print(f"\n[INFO] Initializing {model_type} adaptive system...")
        
        if model_type == 'sklearn':
            # Sklearn model (Online learning destekli)
            base_model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            self.adaptive_system = AdaptiveModelSystem(base_model)
            self.adaptive_system.initial_train(self.X_train_scaled, self.y_train)
            
        elif model_type == 'pytorch':
            # PyTorch model
            input_dim = self.X_train_scaled.shape[1]
            self.pytorch_system = PyTorchAdaptiveSystem(input_dim)
            self.pytorch_system.train_initial(self.X_train_scaled, self.y_train, epochs=30)
            
        # Data drift analyzer
        self.data_drift_analyzer = DataDriftAnalyzer(self.X_train_scaled)
        
        print("[SUCCESS] Systems initialized")
        
    def simulate_data_drift(self, X, drift_type='gradual', intensity=0.3):
        """
        Veri driftini simüle et (Test amaçlı)
        
        Args:
            X: Orijinal veri
            drift_type: 'gradual', 'sudden', 'seasonal'
            intensity: Drift şiddeti (0-1)
        """
        X_drifted = X.copy()
        
        if drift_type == 'gradual':
            # Gradual shift - tüm özelliklere küçük değişim
            noise = np.random.normal(0, intensity, X.shape)
            X_drifted = X_drifted + noise
            
        elif drift_type == 'sudden':
            # Sudden shift - bazı özelliklerde ani değişim
            n_features_to_drift = max(1, int(X.shape[1] * 0.3))
            drift_features = np.random.choice(X.shape[1], n_features_to_drift, replace=False)
            
            for feat_idx in drift_features:
                shift = np.random.uniform(-intensity*2, intensity*2)
                X_drifted.iloc[:, feat_idx] += shift
                
        elif drift_type == 'seasonal':
            # Seasonal pattern - periyodik değişim
            for i in range(X.shape[1]):
                phase = np.random.uniform(0, 2*np.pi)
                seasonal_effect = intensity * np.sin(np.linspace(phase, phase + 4*np.pi, len(X)))
                X_drifted.iloc[:, i] += seasonal_effect
                
        return X_drifted
    
    def run_comprehensive_experiment(self, n_time_periods=10, drift_start=5):
        """
        Kapsamlı deney - Model yaşlanması simülasyonu
        
        Args:
            n_time_periods: Toplam zaman periyodu sayısı
            drift_start: Drift'in başladığı periyot
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL AGING EXPERIMENT")
        print("="*80)
        print(f"Time periods: {n_time_periods}")
        print(f"Drift starts at period: {drift_start}")
        
        # Test verisini parçalara böl
        period_size = len(self.X_test_scaled) // n_time_periods
        
        metrics_history = []
        drift_detected_periods = []
        
        for period in range(n_time_periods):
            print(f"\n{'='*80}")
            print(f"TIME PERIOD {period + 1}/{n_time_periods}")
            print(f"{'='*80}")
            
            # Bu periyot için veriyi al
            start_idx = period * period_size
            end_idx = start_idx + period_size
            
            X_period = self.X_test_scaled.iloc[start_idx:end_idx]
            y_period = self.y_test.iloc[start_idx:end_idx]
            
            # Drift simülasyonu (belirli periyottan sonra)
            if period >= drift_start:
                drift_intensity = 0.1 + (period - drift_start) * 0.05  # Artan drift
                X_period = self.simulate_data_drift(
                    X_period, 
                    drift_type='gradual', 
                    intensity=drift_intensity
                )
                print(f"[SIMULATION] Drift applied (intensity={drift_intensity:.2f})")
            
            # Data drift analizi
            if period > 0:  # İlk periyotta karşılaştırma yok
                print(f"\n--- Data Drift Analysis ---")
                drift_report = self.data_drift_analyzer.comprehensive_drift_report(X_period)
                
                if len(drift_report['drift_features']) > 0:
                    drift_detected_periods.append(period)
            
            # Model performansını değerlendir
            if self.adaptive_system:
                metrics = self.adaptive_system.evaluate_and_record(
                    X_period, y_period, timestamp=period
                )
            elif self.pytorch_system:
                y_pred = self.pytorch_system.predict(X_period)
                metrics = {
                    'accuracy': accuracy_score(y_period, y_pred),
                    'precision': precision_score(y_period, y_pred, zero_division=0),
                    'recall': recall_score(y_period, y_pred, zero_division=0),
                    'f1': f1_score(y_period, y_pred, zero_division=0)
                }
                
            metrics_history.append(metrics)
            
            print(f"\n--- Model Performance ---")
            for metric_name, value in metrics.items():
                print(f"   {metric_name.upper()}: {value:.4f}")
            
            # Degradation detection
            if period >= 3:  # En az 3 periyot gerekli
                if self.adaptive_system:
                    degradation = self.adaptive_system.performance_monitor.detect_degradation(
                        window_size=min(5, period + 1)
                    )
                    
                    print(f"\n--- Degradation Detection ---")
                    for metric_name, result in degradation.items():
                        if result['degradation_detected']:
                            print(f"   [WARNING] {metric_name.upper()} is degrading!")
                            print(f"      Trend slope: {result['trend_slope']:.6f}")
                            print(f"      Change rate: {result['change_rate']*100:.2f}%")
            
            # Adaptive learning (drift tespit edilirse)
            if period in drift_detected_periods:
                print(f"\n[ACTION] Applying adaptive learning...")
                
                if self.adaptive_system:
                    # Incremental update
                    update_size = min(1000, len(X_period))
                    X_update = X_period.iloc[:update_size]
                    y_update = y_period.iloc[:update_size]
                    
                    self.adaptive_system._incremental_update(X_update, y_update)
                    
                elif self.pytorch_system:
                    # PyTorch incremental update
                    update_size = min(1000, len(X_period))
                    X_update = X_period.iloc[:update_size]
                    y_update = y_period.iloc[:update_size]
                    
                    self.pytorch_system.incremental_update(X_update, y_update, epochs=3)
        
        # Final analysis ve visualization
        self._generate_final_report(metrics_history, drift_detected_periods)
        
        return metrics_history
    
    def _generate_final_report(self, metrics_history, drift_detected_periods):
        """Final rapor ve görselleştirmeler"""
        print("\n" + "="*80)
        print("FINAL ANALYSIS REPORT")
        print("="*80)
        
        # 1. Performance trends
        print("\n1. PERFORMANCE TRENDS")
        print("-" * 80)
        
        df_metrics = pd.DataFrame(metrics_history)
        
        for col in df_metrics.columns:
            initial = df_metrics[col].iloc[0]
            final = df_metrics[col].iloc[-1]
            change = ((final - initial) / initial) * 100
            
            print(f"{col.upper()}:")
            print(f"   Initial: {initial:.4f}")
            print(f"   Final: {final:.4f}")
            print(f"   Change: {change:+.2f}%")
        
        # 2. Drift summary
        print(f"\n2. DRIFT DETECTION SUMMARY")
        print("-" * 80)
        print(f"Drift detected in {len(drift_detected_periods)} periods: {drift_detected_periods}")
        
        # 3. Visualizations
        self._create_visualizations(df_metrics, drift_detected_periods)
    
    def _create_visualizations(self, df_metrics, drift_periods):
        """Kapsamlı görselleştirmeler"""
        
        # Figure 1: Performance Metrics Over Time
        fig = plt.figure(figsize=(16, 10))
        
        # Subplot 1: All metrics
        ax1 = plt.subplot(2, 2, 1)
        for col in df_metrics.columns:
            ax1.plot(df_metrics.index, df_metrics[col], marker='o', linewidth=2, label=col.upper())
        
        # Drift periyotlarını işaretle
        for period in drift_periods:
            ax1.axvline(x=period, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Performance degradation rate
        ax2 = plt.subplot(2, 2, 2)
        
        for col in df_metrics.columns:
            # Moving average için degradation rate
            values = df_metrics[col].values
            degradation_rate = np.diff(values) / values[:-1] * 100
            ax2.plot(range(1, len(degradation_rate)+1), degradation_rate, 
                    marker='o', label=col.upper(), alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Time Period', fontsize=12)
        ax2.set_ylabel('Change Rate (%)', fontsize=12)
        ax2.set_title('Performance Change Rate', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Cumulative performance loss
        ax3 = plt.subplot(2, 2, 3)
        
        for col in df_metrics.columns:
            initial_value = df_metrics[col].iloc[0]
            cumulative_loss = ((df_metrics[col] - initial_value) / initial_value) * 100
            ax3.plot(df_metrics.index, cumulative_loss, marker='o', linewidth=2, label=col.upper())
        
        for period in drift_periods:
            ax3.axvline(x=period, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Time Period', fontsize=12)
        ax3.set_ylabel('Cumulative Change (%)', fontsize=12)
        ax3.set_title('Cumulative Performance Change', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Heatmap of metrics
        ax4 = plt.subplot(2, 2, 4)
        
        sns.heatmap(df_metrics.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, ax=ax4)
        ax4.set_xlabel('Time Period', fontsize=12)
        ax4.set_ylabel('Metric', fontsize=12)
        ax4.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_aging_analysis.png', dpi=300, bbox_inches='tight')
        print("\n[INFO] Visualization saved as 'model_aging_analysis.png'")
        plt.show()
        
        # Figure 2: Advanced drift analysis
        if len(drift_periods) > 0:
            self._create_drift_visualization(drift_periods)
    
    def _create_drift_visualization(self, drift_periods):
        """Drift analizi görselleştirmesi"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Drift timeline
        ax1 = axes[0, 0]
        periods = list(range(len(self.X_test_scaled) // 10))
        drift_indicator = [1 if p in drift_periods else 0 for p in periods]
        
        ax1.bar(periods, drift_indicator, color=['red' if d else 'green' for d in drift_indicator])
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Drift Detected')
        ax1.set_title('Drift Detection Timeline')
        ax1.set_ylim([0, 1.2])
        
        # Subplot 2: Feature importance change (simulated)
        ax2 = axes[0, 1]
        
        if hasattr(self.adaptive_system, 'current_model'):
            if hasattr(self.adaptive_system.current_model, 'coef_'):
                feature_importance = np.abs(self.adaptive_system.current_model.coef_[0])
                top_features = np.argsort(feature_importance)[-10:]
                
                ax2.barh(range(len(top_features)), feature_importance[top_features])
                ax2.set_xlabel('Importance')
                ax2.set_ylabel('Feature Index')
                ax2.set_title('Top 10 Feature Importances')
        
        # Subplot 3: Distribution comparison
        ax3 = axes[1, 0]
        
        # İlk ve son periyot karşılaştırması
        period_size = len(self.X_test_scaled) // 10
        X_first = self.X_test_scaled.iloc[:period_size]
        X_last = self.X_test_scaled.iloc[-period_size:]
        
        feature_to_plot = self.X_test_scaled.columns[0]  # İlk özellik
        
        ax3.hist(X_first[feature_to_plot], bins=50, alpha=0.5, label='Initial', density=True)
        ax3.hist(X_last[feature_to_plot], bins=50, alpha=0.5, label='Final', density=True)
        ax3.set_xlabel(f'Feature: {feature_to_plot}')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Shift Example')
        ax3.legend()
        
        # Subplot 4: Drift severity over time
        ax4 = axes[1, 1]
        
        # Simulated drift severity
        drift_severity = np.zeros(len(periods))
        for p in drift_periods:
            if p < len(drift_severity):
                drift_severity[p] = np.random.uniform(0.3, 0.9)
        
        ax4.plot(periods, drift_severity, marker='o', linewidth=2, color='red')
        ax4.fill_between(periods, 0, drift_severity, alpha=0.3, color='red')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Drift Severity')
        ax4.set_title('Drift Severity Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drift_analysis.png', dpi=300, bbox_inches='tight')
        print("[INFO] Drift visualization saved as 'drift_analysis.png'")
        plt.show()


# ============================================================================
# 7. DEMO / KULLANIM ÖRNEĞİ
# ============================================================================

def run_demo():
    """
    Demo - Credit Card Fraud veri seti ile kullanım örneği
    """
    print("\n" + "="*80)
    print("DEMO: CREDIT CARD FRAUD DETECTION - MODEL AGING ANALYSIS")
    print("="*80)
    
    # NOT: Kullanıcı kendi veri setini yüklemeli
    print("\n[INFO] To use this system with your Credit Card Fraud dataset:")
    print("1. Download data from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Place 'creditcard.csv' in your working directory")
    print("3. Run the code below:\n")
    
    print("# Example usage:")
    print("""
# Sistemi başlat
manager = ModelLifecycleManager(data_path='creditcard.csv')

# Veriyi hazırla
manager.prepare_data(target_column='Class', test_size=0.3)

# Sklearn model ile sistem
manager.initialize_systems(model_type='sklearn')

# Kapsamlı deney çalıştır
metrics_history = manager.run_comprehensive_experiment(
    n_time_periods=10,
    drift_start=5
)

# Alternatif: PyTorch model kullan
# manager.initialize_systems(model_type='pytorch')
# metrics_history = manager.run_comprehensive_experiment(n_time_periods=10, drift_start=5)
""")
    
    print("\n[INFO] System is ready! Load your data and start the experiment.")
    print("\nFeatures:")
    print("  ✓ Real-time drift detection (ADWIN, Page-Hinkley, DDM)")
    print("  ✓ Data drift analysis (KS test, PSI, Wasserstein distance)")
    print("  ✓ Performance degradation prediction")
    print("  ✓ Adaptive learning (Incremental & online learning)")
    print("  ✓ Multi-model support (Sklearn & PyTorch)")
    print("  ✓ Comprehensive visualizations")
    print("  ✓ Automated reporting")
    
    print("\n" + "="*80)


# ============================================================================
# 8. UTILITY FUNCTIONS
# ============================================================================

def compare_models_over_time(model_managers, labels, n_periods=10):
    """
    Birden fazla modeli karşılaştır
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for manager, label in zip(model_managers, labels):
            if hasattr(manager, 'adaptive_system'):
                history = manager.adaptive_system.performance_monitor.performance_history
                values = [h[metric] for h in history]
                ax.plot(range(len(values)), values, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def export_drift_report(manager, output_path='drift_report.txt'):
    """
    Drift raporunu dosyaya kaydet
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL DRIFT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        if hasattr(manager, 'adaptive_system') and manager.adaptive_system:
            history = manager.adaptive_system.performance_monitor.performance_history
            
            f.write("Performance History:\n")
            f.write("-"*80 + "\n")
            
            for idx, metrics in enumerate(history):
                f.write(f"\nPeriod {idx + 1}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated successfully!\n")
    
    print(f"[INFO] Report exported to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MODEL AGING DETECTION SYSTEM" + " "*30 + "║")
    print("║" + " "*78 + "║")
    print("║" + "  Comprehensive ML Model Lifecycle Management & Drift Detection" + " "*14 + "║")
    print("╚" + "="*78 + "╝")
    
    run_demo()
    
    print("\n[SYSTEM READY] You can now:")
    print("  1. Load your Credit Card Fraud dataset")
    print("  2. Create ModelLifecycleManager instance")
    print("  3. Run comprehensive experiments")
    print("  4. Analyze model aging patterns")
    print("  5. Apply adaptive learning strategies")
    
    print("\n" + "="*80)
    print("Happy Model Monitoring! 🚀")
    print("="*80 + "\n")