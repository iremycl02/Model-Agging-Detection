
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from io import BytesIO
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from scipy import stats

# Page config
st.set_page_config(
    page_title="Model Aging Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_drift(X, intensity=0.3, drift_type='gradual'):
    """Drift simülasyonu"""
    X_drifted = X.copy()
    
    if drift_type == 'gradual':
        noise = np.random.normal(0, intensity, X.shape)
        X_drifted = X_drifted + noise
    elif drift_type == 'sudden':
        n_features = max(1, int(X.shape[1] * 0.3))
        features = np.random.choice(X.shape[1], n_features, replace=False)
        for feat_idx in features:
            shift = np.random.uniform(-intensity*2, intensity*2)
            X_drifted.iloc[:, feat_idx] += shift
    elif drift_type == 'seasonal':
        for i in range(X.shape[1]):
            phase = np.random.uniform(0, 2*np.pi)
            seasonal = intensity * np.sin(np.linspace(phase, phase + 4*np.pi, len(X)))
            X_drifted.iloc[:, i] += seasonal
    
    return X_drifted


def create_plotly_performance_chart(df_metrics, problem_type):
    """Plotly ile performans grafiği"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Over Time', 'Drift Timeline', 
                       'Performance Change Rate', 'Cumulative Change'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Metrikleri belirle
    if problem_type in ['binary', 'classification']:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    else:
        metrics = ['r2', 'mae', 'rmse']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # 1. Performance Over Time
    for metric, color in zip(metrics, colors):
        if metric in df_metrics.columns:
            fig.add_trace(
                go.Scatter(x=df_metrics['period'], y=df_metrics[metric],
                          name=metric.upper(), mode='lines+markers',
                          line=dict(color=color, width=2),
                          marker=dict(size=8)),
                row=1, col=1
            )
    
    # 2. Drift Timeline
    drift_colors = ['red' if d else 'green' for d in df_metrics['drift_applied']]
    fig.add_trace(
        go.Bar(x=df_metrics['period'], y=[1]*len(df_metrics),
              marker_color=drift_colors, showlegend=False),
        row=1, col=2
    )
    
    # 3. Change Rate
    main_metric = metrics[0]
    values = df_metrics[main_metric].values
    change_rate = np.diff(values) / values[:-1] * 100
    fig.add_trace(
        go.Scatter(x=df_metrics['period'][1:], y=change_rate,
                  name='Change Rate', mode='lines+markers',
                  line=dict(color='#e74c3c', width=2)),
        row=2, col=1
    )
    
    # 4. Cumulative Change
    initial = values[0]
    cumulative = ((values - initial) / initial) * 100
    fig.add_trace(
        go.Scatter(x=df_metrics['period'], y=cumulative,
                  name='Cumulative', mode='lines+markers',
                  fill='tozeroy', line=dict(color='#9b59b6', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=True, title_text="Model Performance Analysis")
    return fig


def generate_report_html(df_metrics, problem_type, model_name):
    """HTML rapor oluştur"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Ana metrik
    main_metric = 'f1' if problem_type in ['binary', 'classification'] else 'r2'
    initial = df_metrics[main_metric].iloc[0]
    final = df_metrics[main_metric].iloc[-1]
    change = ((final - initial) / initial) * 100
    
    # Status belirleme
    if final < initial * 0.90:
        status = "🔴 CRITICAL"
        status_color = "#dc3545"
    elif final < initial * 0.95:
        status = "🟡 WARNING"
        status_color = "#ffc107"
    else:
        status = "🟢 HEALTHY"
        status_color = "#28a745"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; text-align: center; border-radius: 10px; }}
            .content {{ background-color: white; padding: 20px; margin-top: 20px; 
                       border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 15px; padding: 15px; 
                      background-color: #f8f9fa; border-radius: 8px; min-width: 200px; }}
            .metric-value {{ font-size: 28px; font-weight: bold; color: {status_color}; }}
            .status {{ background-color: {status_color}; color: white; padding: 10px; 
                      border-radius: 5px; display: inline-block; margin: 10px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #667eea; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Model Aging Analysis Report</h1>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="content">
            <h2>📈 Executive Summary</h2>
            <div class="status">{status}</div>
            
            <div class="metric">
                <div style="color: #666;">Model</div>
                <div class="metric-value" style="font-size: 20px; color: #333;">{model_name}</div>
            </div>
            
            <div class="metric">
                <div style="color: #666;">Initial {main_metric.upper()}</div>
                <div class="metric-value">{initial:.4f}</div>
            </div>
            
            <div class="metric">
                <div style="color: #666;">Final {main_metric.upper()}</div>
                <div class="metric-value">{final:.4f}</div>
            </div>
            
            <div class="metric">
                <div style="color: #666;">Change</div>
                <div class="metric-value">{change:+.2f}%</div>
            </div>
            
            <h2>📊 Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Period</th>
                    {''.join([f'<th>{col.upper()}</th>' for col in df_metrics.columns if col not in ['period', 'drift_applied']])}
                    <th>Drift</th>
                </tr>
    """
    
    for _, row in df_metrics.iterrows():
        html += "<tr>"
        html += f"<td>{row['period']}</td>"
        for col in df_metrics.columns:
            if col not in ['period', 'drift_applied']:
                html += f"<td>{row[col]:.4f}</td>"
        html += f"<td>{'🔴 Yes' if row['drift_applied'] else '🟢 No'}</td>"
        html += "</tr>"
    
    html += """
            </table>
            
            <h2>💡 Recommendations</h2>
    """
    
    if final < initial * 0.90:
        html += """
            <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ CRITICAL:</strong> Performance dropped more than 10%<br>
                <strong>Action:</strong> Immediate model retraining required
            </div>
        """
    elif final < initial * 0.95:
        html += """
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ WARNING:</strong> Performance dropped 5-10%<br>
                <strong>Action:</strong> Consider incremental learning or fine-tuning
            </div>
        """
    else:
        html += """
            <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>✅ HEALTHY:</strong> Performance is acceptable<br>
                <strong>Action:</strong> Continue monitoring
            </div>
        """
    
    drift_count = df_metrics['drift_applied'].sum()
    if drift_count > len(df_metrics) * 0.3:
        html += """
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ High Drift Frequency:</strong> Data distribution is changing<br>
                <strong>Action:</strong> Implement continuous learning strategy
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Model Aging Detection</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("⚙️ Configuration")
        
        # Ana akış adımları
        st.markdown("### 📋 Workflow Steps")
        steps = {
            "1️⃣ Data Loading": st.session_state.data_loaded,
            "2️⃣ Model Setup": st.session_state.model_trained,
            "3️⃣ Analysis": st.session_state.analysis_done
        }
        
        for step, completed in steps.items():
            if completed:
                st.success(f"{step} ✅")
            else:
                st.info(f"{step} ⏳")
        
        st.markdown("---")
        
        # Problem type selection
        problem_type = st.selectbox(
            "Problem Type",
            ["binary", "classification", "regression"],
            help="Select your ML problem type"
        )
        
        st.markdown("---")
        
        # Quick Actions
        st.info("""
    **Metric Descriptions:**
    
    📊 **MDI**: Model Degradation Index
    - 0-0.3: Healthy
    - 0.3-0.5: Attention
    - 0.5+: Critical
    
    🎯 **AUC**: Model performance
    - 0.9+: Perfect
    - 0.85-0.9: Good
    - <0.85: Weak
    """)
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.button("📖 Documentation"):
            st.info("Check the README.md for detailed documentation")
    
    # Main content
    tabs = st.tabs(["📁 Data & Model", "🔬 Analysis", "📊 Results", "📄 Reports"])
    
    # ========================================================================
    # TAB 1: DATA & MODEL
    # ========================================================================
    with tabs[0]:
        st.header("📁 Data & Model Setup")
        
        col1, col2 = st.columns(2)
        
        # DATA LOADING
        with col1:
            st.subheader("1️⃣ Load Data")
            
            data_source = st.radio(
                "Data Source",
                ["Upload CSV/Excel", "Use Sample Data"],
                horizontal=True
            )
            
            if data_source == "Upload CSV/Excel":
                uploaded_file = st.file_uploader(
                    "Choose a file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload your dataset"
                )
                
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                        
                        # Preview
                        with st.expander("👀 Preview Data"):
                            st.dataframe(df.head())
                            
                            # Basic stats
                            st.markdown("**Basic Statistics:**")
                            st.write(df.describe())
                        
                        # Target column selection
                        target_col = st.selectbox(
                            "Select Target Column",
                            df.columns.tolist(),
                            help="Choose the column you want to predict"
                        )
                        st.session_state.target_col = target_col
                        
                        # Show target distribution
                        if target_col:
                            st.markdown("**Target Distribution:**")
                            if problem_type in ['binary', 'classification']:
                                target_counts = df[target_col].value_counts()
                                fig = px.bar(x=target_counts.index, y=target_counts.values,
                                           labels={'x': target_col, 'y': 'Count'},
                                           color=target_counts.values,
                                           color_continuous_scale='viridis')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = px.histogram(df, x=target_col, nbins=50)
                                st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            
            else:  # Sample data
                st.info("Generating sample data...")
                from sklearn.datasets import make_classification, make_regression
                
                if problem_type in ['binary', 'classification']:
                    X, y = make_classification(
                        n_samples=5000, n_features=15, n_informative=10,
                        n_classes=2, weights=[0.8, 0.2], random_state=42
                    )
                    feature_names = [f'feature_{i}' for i in range(15)]
                else:
                    X, y = make_regression(
                        n_samples=5000, n_features=15, n_informative=10,
                        noise=10, random_state=42
                    )
                    feature_names = [f'feature_{i}' for i in range(15)]
                
                df = pd.DataFrame(X, columns=feature_names)
                df['target'] = y
                
                st.session_state.df = df
                st.session_state.target_col = 'target'
                st.session_state.data_loaded = True
                
                st.success(f"✅ Sample data generated: {df.shape[0]} rows × {df.shape[1]} columns")
                
                with st.expander("👀 Preview Sample Data"):
                    st.dataframe(df.head())
        
        # MODEL SETUP
        with col2:
            st.subheader("2️⃣ Model Setup")
            
            if not st.session_state.data_loaded:
                st.warning("⚠️ Please load data first")
            else:
                model_source = st.radio(
                    "Model Source",
                    ["Upload Trained Model", "Train New Model"],
                    horizontal=True
                )
                
                if model_source == "Upload Trained Model":
                    uploaded_model = st.file_uploader(
                        "Upload Model (.pkl or .joblib)",
                        type=['pkl', 'joblib'],
                        help="Upload your trained model"
                    )
                    
                    if uploaded_model:
                        try:
                            if uploaded_model.name.endswith('.pkl'):
                                model = pickle.load(uploaded_model)
                            else:
                                model = joblib.load(uploaded_model)
                            
                            st.session_state.model = model
                            st.session_state.model_trained = True
                            st.success(f"✅ Model loaded: {type(model).__name__}")
                            
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
                
                else:  # Train new model
                    st.markdown("**Model Configuration:**")
                    
                    if problem_type in ['binary', 'classification']:
                        model_type = st.selectbox(
                            "Select Model",
                            ["Random Forest", "Gradient Boosting", "Logistic Regression", "SGD Classifier"]
                        )
                    else:
                        model_type = st.selectbox(
                            "Select Model",
                            ["Random Forest", "Gradient Boosting"]
                        )
                    
                    # Hyperparameters
                    with st.expander("⚙️ Hyperparameters"):
                        if "Random Forest" in model_type:
                            n_estimators = st.slider("n_estimators", 50, 300, 100)
                            max_depth = st.slider("max_depth", 3, 20, 10)
                        elif "Gradient Boosting" in model_type:
                            n_estimators = st.slider("n_estimators", 50, 200, 100)
                            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1)
                    
                    # Scaling
                    scaler_type = st.selectbox(
                        "Data Scaling",
                        ["robust", "standard", "minmax", "none"]
                    )
                    
                    if st.button("🚀 Train Model", type="primary"):
                        with st.spinner("Training model..."):
                            try:
                                df = st.session_state.df
                                target_col = st.session_state.target_col
                                
                                # Split data
                                X = df.drop(columns=[target_col])
                                y = df[target_col]
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=42,
                                    stratify=y if problem_type in ['binary', 'classification'] else None
                                )
                                
                                # Scaling
                                if scaler_type != 'none':
                                    if scaler_type == 'standard':
                                        scaler = StandardScaler()
                                    elif scaler_type == 'robust':
                                        scaler = RobustScaler()
                                    else:
                                        scaler = MinMaxScaler()
                                    
                                    X_train = pd.DataFrame(
                                        scaler.fit_transform(X_train),
                                        columns=X_train.columns
                                    )
                                    X_test = pd.DataFrame(
                                        scaler.transform(X_test),
                                        columns=X_test.columns
                                    )
                                    st.session_state.scaler = scaler
                                
                                # Model creation
                                if model_type == "Random Forest":
                                    if problem_type in ['binary', 'classification']:
                                        model = RandomForestClassifier(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42,
                                            n_jobs=-1
                                        )
                                    else:
                                        model = RandomForestRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42,
                                            n_jobs=-1
                                        )
                                elif model_type == "Gradient Boosting":
                                    if problem_type in ['binary', 'classification']:
                                        model = GradientBoostingClassifier(
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            random_state=42
                                        )
                                    else:
                                        from sklearn.ensemble import GradientBoostingRegressor
                                        model = GradientBoostingRegressor(
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            random_state=42
                                        )
                                elif model_type == "Logistic Regression":
                                    model = LogisticRegression(random_state=42, max_iter=1000)
                                else:  # SGD
                                    model = SGDClassifier(random_state=42, max_iter=1000)
                                
                                # Train
                                model.fit(X_train, y_train)
                                
                                # Store
                                st.session_state.model = model
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.session_state.model_trained = True
                                
                                # Evaluate
                                y_pred = model.predict(X_test)
                                
                                st.success("✅ Model trained successfully!")
                                
                                # Show initial performance
                                st.markdown("**Initial Performance:**")
                                if problem_type in ['binary', 'classification']:
                                    col_a, col_b, col_c = st.columns(3)
                                    col_a.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                                    col_b.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                                    col_c.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                                else:
                                    col_a, col_b = st.columns(2)
                                    col_a.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                                    col_b.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                                
                            except Exception as e:
                                st.error(f"Error during training: {str(e)}")
    
    # ========================================================================
    # TAB 2: ANALYSIS
    # ========================================================================
    with tabs[1]:
        st.header("🔬 Model Aging Analysis")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train or upload a model first")
        else:
            st.success("✅ Ready for analysis")
            
            # Analysis parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Simulation Parameters")
                n_periods = st.slider(
                    "Number of Time Periods",
                    min_value=5, max_value=30, value=10,
                    help="How many time periods to simulate"
                )
                
                drift_start = st.slider(
                    "Drift Start Period",
                    min_value=1, max_value=n_periods-1, value=min(5, n_periods-1),
                    help="When should drift begin"
                )
                
                drift_intensity = st.slider(
                    "Drift Intensity",
                    min_value=0.05, max_value=0.5, value=0.15, step=0.05,
                    help="How strong should the drift be"
                )
            
            with col2:
                st.subheader("🎨 Drift Configuration")
                drift_type = st.selectbox(
                    "Drift Type",
                    ["gradual", "sudden", "seasonal"],
                    help="Type of data drift to simulate"
                )
                
                enable_adaptation = st.checkbox(
                    "Enable Model Adaptation",
                    value=False,
                    help="Allow incremental learning (only for compatible models)"
                )
                
                st.markdown("**Drift Type Explanation:**")
                if drift_type == "gradual":
                    st.info("📈 Gradual: Slow, continuous changes over time")
                elif drift_type == "sudden":
                    st.info("⚡ Sudden: Abrupt changes in random features")
                else:
                    st.info("🌊 Seasonal: Periodic, cyclical patterns")
            
            # Run analysis button
            if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Running aging analysis..."):
                    try:
                        model = st.session_state.model
                        X_test = st.session_state.X_test
                        y_test = st.session_state.y_test
                        
                        period_size = len(X_test) // n_periods
                        metrics_history = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Analysis loop
                        for period in range(n_periods):
                            status_text.text(f"Analyzing period {period + 1}/{n_periods}...")
                            progress_bar.progress((period + 1) / n_periods)
                            
                            # Get period data
                            start_idx = period * period_size
                            end_idx = start_idx + period_size
                            
                            X_period = X_test.iloc[start_idx:end_idx].copy()
                            y_period = y_test.iloc[start_idx:end_idx]
                            
                            # Apply drift
                            drift_applied = False
                            if period >= drift_start:
                                intensity = drift_intensity + (period - drift_start) * 0.05
                                X_period = simulate_drift(X_period, intensity, drift_type)
                                drift_applied = True
                            
                            # Predict
                            y_pred = model.predict(X_period)
                            
                            # Calculate metrics
                            metrics = {'period': period + 1, 'drift_applied': drift_applied}
                            
                            if problem_type in ['binary', 'classification']:
                                metrics['accuracy'] = accuracy_score(y_period, y_pred)
                                metrics['precision'] = precision_score(y_period, y_pred, average='weighted', zero_division=0)
                                metrics['recall'] = recall_score(y_period, y_pred, average='weighted', zero_division=0)
                                metrics['f1'] = f1_score(y_period, y_pred, average='weighted', zero_division=0)
                                
                                if hasattr(model, 'predict_proba'):
                                    y_proba = model.predict_proba(X_period)[:, 1] if len(np.unique(y_period)) == 2 else None
                                    if y_proba is not None:
                                        metrics['auc_roc'] = roc_auc_score(y_period, y_proba)
                            else:
                                metrics['r2'] = r2_score(y_period, y_pred)
                                metrics['mae'] = mean_absolute_error(y_period, y_pred)
                                metrics['rmse'] = np.sqrt(mean_squared_error(y_period, y_pred))
                            
                            metrics_history.append(metrics)
                            
                            # Adaptation (if enabled and supported)
                            if enable_adaptation and drift_applied and hasattr(model, 'partial_fit'):
                                update_size = min(500, len(X_period))
                                model.partial_fit(X_period.iloc[:update_size], y_period.iloc[:update_size])
                        
                        # Store results
                        st.session_state.metrics_history = metrics_history
                        st.session_state.analysis_done = True
                        st.session_state.problem_type = problem_type
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Analysis complete!")
                        
                        st.success("🎉 Analysis completed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 3: RESULTS
    # ========================================================================
    with tabs[2]:
        st.header("Analysis Results")
        
        if not st.session_state.analysis_done:
            st.info("ℹ️ Run analysis first to see results")
        else:
            df_metrics = pd.DataFrame(st.session_state.metrics_history)
            problem_type = st.session_state.problem_type
            
            # MODEL PERFORMANS ÖZETİ TABLOSU
            st.markdown("### 📊 Model Performance Summary")
            
            # Tablo verileri hazırla
            model_name = type(st.session_state.model).__name__
            
            if problem_type in ['binary', 'classification'] and 'auc_roc' in df_metrics.columns:
                # İlk ve son değerleri al
                initial_auc = df_metrics['auc_roc'].iloc[0]
                final_auc = df_metrics['auc_roc'].iloc[-1]
                avg_auc = df_metrics['auc_roc'].mean()
                
                initial_mdi = 0.0
                final_mdi = abs((final_auc - initial_auc) / initial_auc)
                avg_mdi = np.mean([abs((df_metrics['auc_roc'].iloc[i] - initial_auc) / initial_auc) 
                                  for i in range(len(df_metrics))])
                
                # Sağlık skoru hesapla
                health_score = (final_auc / initial_auc) * 100
                
                # Yaşlanma hızı hesapla (period başına)
                aging_rate = final_mdi / len(df_metrics)
                
                # Performans tablosu
                perf_data = {
                    'Model': [model_name],
                    'Health Score': [f'{health_score:.1f}/100'],
                    'Final AUC': [f'{final_auc:.4f}'],
                    'Avg AUC': [f'{avg_auc:.4f}'],
                    'Final MDI': [f'{final_mdi:.4f}'],
                    'Avg MDI': [f'{avg_mdi:.4f}'],
                    'Ageing Rate': [f'{aging_rate:.4f}/period']
                }
                
                perf_df = pd.DataFrame(perf_data)
                
                # Tabloyu göster
                st.dataframe(
                    perf_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # SKOR KARTLARI
                st.markdown("### 📈 Current Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Skor hesaplamaları
                main_metric = 'f1' if problem_type in ['binary', 'classification'] else 'r2'
                initial_main = df_metrics[main_metric].iloc[0]
                final_main = df_metrics[main_metric].iloc[-1]
                change_main = final_main - initial_main
                
                mdi_change = final_mdi - initial_mdi
                auc_change = final_auc - initial_auc
                
                # Güven skoru (confidence)
                confidence = 100 - (final_mdi * 100)
                
                # Kart 1: Model Sağlık Skoru
                with col1:
                    st.markdown("""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #9b59b6;'>
                        <p style='color: #888; font-size: 14px; margin: 0;'>📊 Model Health Score</p>
                        <h2 style='margin: 10px 0; color: white;'>{:.1f}/100</h2>
                        <p style='color: {}; font-size: 14px; margin: 0;'>{} {:.1f}</p>
                    </div>
                    """.format(
                        health_score,
                        '#e74c3c' if change_main < 0 else '#2ecc71',
                        '↓' if change_main < 0 else '↑',
                        abs(change_main * 100)
                    ), unsafe_allow_html=True)
                
                # Kart 2: Güncel MDI
                with col2:
                    st.markdown("""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #e74c3c;'>
                        <p style='color: #888; font-size: 14px; margin: 0;'>📈 Current MDI</p>
                        <h2 style='margin: 10px 0; color: white;'>{:.3f}</h2>
                        <p style='color: {}; font-size: 14px; margin: 0;'>{} {:.4f}</p>
                    </div>
                    """.format(
                        final_mdi,
                        '#e74c3c' if mdi_change > 0 else '#2ecc71',
                        '↑' if mdi_change > 0 else '↓',
                        abs(mdi_change)
                    ), unsafe_allow_html=True)
                
                # Kart 3: Güncel AUC
                with col3:
                    st.markdown("""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;'>
                        <p style='color: #888; font-size: 14px; margin: 0;'>🎯 Current AUC</p>
                        <h2 style='margin: 10px 0; color: white;'>{:.4f}</h2>
                        <p style='color: {}; font-size: 14px; margin: 0;'>{} {:.4f}</p>
                    </div>
                    """.format(
                        final_auc,
                        '#2ecc71' if auc_change > 0 else '#e74c3c',
                        '↑' if auc_change > 0 else '↓',
                        abs(auc_change)
                    ), unsafe_allow_html=True)
                
                # Kart 4: Güven
                with col4:
                    st.markdown("""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #2ecc71;'>
                        <p style='color: #888; font-size: 14px; margin: 0;'>🔒 Trust</p>
                        <h2 style='margin: 10px 0; color: white;'>{:.2f}%</h2>
                        <p style='color: {}; font-size: 14px; margin: 0;'>{}</p>
                    </div>
                    """.format(
                        confidence,
                        '#2ecc71' if confidence > 80 else '#e74c3c',
                        '↑ Low' if confidence > 80 else '↓ High'
                    ), unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Summary metrics
            st.subheader("📈 Performance Summary")
            
            main_metric = 'f1' if problem_type in ['binary', 'classification'] else 'r2'
            initial = df_metrics[main_metric].iloc[0]
            final = df_metrics[main_metric].iloc[-1]
            mean_val = df_metrics[main_metric].mean()
            change = ((final - initial) / initial) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                label=f"Initial {main_metric.upper()}",
                value=f"{initial:.4f}"
            )
            
            col2.metric(
                label=f"Final {main_metric.upper()}",
                value=f"{final:.4f}",
                delta=f"{change:.2f}%"
            )
            
            col3.metric(
                label=f"Average {main_metric.upper()}",
                value=f"{mean_val:.4f}"
            )
            
            drift_count = df_metrics['drift_applied'].sum()
            col4.metric(
                label="Drift Periods",
                value=f"{drift_count}/{len(df_metrics)}"
            )
            
            # Status indicator
            st.markdown("---")
            if final < initial * 0.90:
                st.markdown('<div class="danger-box"><strong>🔴 CRITICAL:</strong> Performance dropped more than 10%. Immediate action required!</div>', unsafe_allow_html=True)
            elif final < initial * 0.95:
                st.markdown('<div class="warning-box"><strong>🟡 WARNING:</strong> Performance dropped 5-10%. Consider model adaptation.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><strong>🟢 HEALTHY:</strong> Performance is within acceptable range.</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Interactive plots
            st.subheader("📉 Interactive Visualizations")
            
            # Main performance chart
            fig = create_plotly_performance_chart(df_metrics, problem_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Ek Grafikler - AUC ve MDI Analizi (Sadece Classification için)
            if problem_type in ['binary', 'classification'] and 'auc_roc' in df_metrics.columns:
                st.subheader("📈 Advanced Performance Metrics")
                
                # AUC ve MDI hesapla
                periods = df_metrics['period'].values
                auc_values = df_metrics['auc_roc'].values
                
                # MDI hesaplama (baseline'dan sapma)
                baseline_auc = auc_values[0]
                mdi_values = np.abs((auc_values - baseline_auc) / baseline_auc)
                
                # 2 grafik yan yana
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. AUC Over Time
                    fig_auc = go.Figure()
                    
                    fig_auc.add_trace(go.Scatter(
                        x=periods,
                        y=auc_values,
                        mode='lines+markers',
                        name='AUC Score',
                        line=dict(color='#00d4ff', width=3),
                        marker=dict(size=10, color='#00d4ff')
                    ))
                    
                    # Warning threshold
                    fig_auc.add_hline(
                        y=0.85, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Warning Threshold",
                        annotation_position="right"
                    )
                    
                    fig_auc.update_layout(
                        title="Model Performance (AUC) Over Time",
                        xaxis_title="Time Period",
                        yaxis_title="AUC Score",
                        template="plotly_dark",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_auc, use_container_width=True)
                
                with col2:
                    # 2. MDI Over Time
                    fig_mdi = go.Figure()
                    
                    fig_mdi.add_trace(go.Scatter(
                        x=periods,
                        y=mdi_values,
                        mode='lines+markers',
                        name='MDI',
                        line=dict(color='#ff00ff', width=3),
                        marker=dict(size=10, color='#ff00ff'),
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 255, 0.1)'
                    ))
                    
                    # Critical threshold
                    fig_mdi.add_hline(
                        y=0.5, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Critical Threshold",
                        annotation_position="right"
                    )
                    
                    fig_mdi.update_layout(
                        title="Model Degradation Index (MDI) Over Time",
                        xaxis_title="Time Period",
                        yaxis_title="MDI",
                        template="plotly_dark",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_mdi, use_container_width=True)
                
                # 3. MDI vs AUC Correlation (Scatter plot)
                st.markdown("### 🔗 MDI vs AUC Correlation")
                
                fig_correlation = go.Figure()
                
                fig_correlation.add_trace(go.Scatter(
                    x=mdi_values,
                    y=auc_values,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=periods,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Period"),
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Period {p}" for p in periods],
                    hovertemplate='<b>Period %{text}</b><br>' +
                                  'MDI: %{x:.4f}<br>' +
                                  'AUC: %{y:.4f}<br>' +
                                  '<extra></extra>'
                ))
                
                fig_correlation.update_layout(
                    title="MDI vs AUC Correlation",
                    xaxis_title="Model Degradation Index (MDI)",
                    yaxis_title="Performance (AUC)",
                    template="plotly_dark",
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_correlation, use_container_width=True)
                
                # Korelasyon istatistiği
                correlation = np.corrcoef(mdi_values, auc_values)[0, 1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MDI-AUC Correlation", f"{correlation:.4f}")
                col2.metric("Max MDI", f"{mdi_values.max():.4f}")
                col3.metric("Min AUC", f"{auc_values.min():.4f}")
                
                if correlation < -0.7:
                    st.warning("⚠️ Strong negative correlation detected: Model performance decreases as degradation increases.")
                elif correlation < -0.4:
                    st.info("ℹ️ Moderate negative correlation: Some performance loss due to degradation.")
                else:
                    st.success("✅ Weak correlation: Model is relatively stable despite degradation.")
            
            st.markdown("---")
            
            # Detaylı Durum Raporu
            st.subheader("📋 Detailed Status Report")
            
            # Metrikler tablosu
            report_data = []
            
            # Model Sağlık Skoru
            health_score = final / initial * 100
            health_status = "🟢 Low" if health_score >= 97 else ("🟡 Medium" if health_score >= 90 else "🔴 High")
            report_data.append({
                "Metric": "Model Health Score",
                "Value": f"{health_score:.1f}/100",
                "Status": health_status
            })
            
            # Güncel MDI
            current_mdi = abs((final - initial) / initial)
            mdi_status = "🔴 High" if current_mdi > 0.1 else ("🟡 Medium" if current_mdi > 0.05 else "🟢 Low")
            report_data.append({
                "Metric": "Current MDI",
                "Value": f"{current_mdi:.4f}",
                "Status": mdi_status
            })
            
            # Güncel AUC (eğer classification ise)
            if problem_type in ['binary', 'classification'] and 'auc_roc' in df_metrics.columns:
                current_auc = df_metrics['auc_roc'].iloc[-1]
                auc_status = "🟢 High" if current_auc >= 0.85 else "🟡 Medium"
                report_data.append({
                    "Metric": "Current AUC",
                    "Value": f"{current_auc:.4f}",
                    "Status": auc_status
                })
            
            # Ortalama MDI
            avg_mdi = np.mean([abs((df_metrics[main_metric].iloc[i] - df_metrics[main_metric].iloc[0]) / 
                                    df_metrics[main_metric].iloc[0]) 
                              for i in range(len(df_metrics))])
            avg_mdi_status = "🟢 Low" if avg_mdi < 0.05 else ("🟡 Medium" if avg_mdi < 0.15 else "🔴 High")
            report_data.append({
                "Metric": "Average MDI",
                "Value": f"{avg_mdi:.4f}",
                "Status": avg_mdi_status
            })
            
            # Ortalama AUC (eğer classification ise)
            if problem_type in ['binary', 'classification'] and 'auc_roc' in df_metrics.columns:
                avg_auc = df_metrics['auc_roc'].mean()
                avg_auc_status = "🟢 Highh" if avg_auc >= 0.85 else "🟡 Medium"
                report_data.append({
                    "Metric": "Average AUC",
                    "Value": f"{avg_auc:.4f}",
                    "Status": avg_auc_status
                })
            
            # Confidence
            confidence = 100 - (current_mdi * 100)
            confidence_status = "🟢 Low" if confidence > 80 else "🔴 High"
            report_data.append({
                "Metric": "Confidence",
                "Value": f"{confidence:.2f}%",
                "Status": confidence_status
            })
            
            # Tabloyu göster
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics Table")
            
            # Format dataframe
            df_display = df_metrics.copy()
            df_display['drift_applied'] = df_display['drift_applied'].apply(lambda x: '🔴 Yes' if x else '🟢 No')
            
            st.dataframe(df_display, use_container_width=True)
            
            # Additional insights
            st.subheader("🔍 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Trend:**")
                
                # Calculate trend
                values = df_metrics[main_metric].values
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                if slope < -0.01:
                    st.error(f"📉 Declining trend (slope: {slope:.6f})")
                    st.write("The model performance is consistently decreasing over time.")
                elif slope > 0.01:
                    st.success(f"📈 Improving trend (slope: {slope:.6f})")
                    st.write("The model performance is improving, possibly due to adaptation.")
                else:
                    st.info(f"➡️ Stable trend (slope: {slope:.6f})")
                    st.write("The model performance is relatively stable.")
            
            with col2:
                st.markdown("**Drift Impact:**")
                
                if drift_count > 0:
                    pre_drift = df_metrics[~df_metrics['drift_applied']][main_metric].mean()
                    post_drift = df_metrics[df_metrics['drift_applied']][main_metric].mean()
                    drift_impact = ((post_drift - pre_drift) / pre_drift) * 100
                    
                    st.metric(
                        label="Drift Impact on Performance",
                        value=f"{drift_impact:+.2f}%",
                        delta=f"{post_drift:.4f} vs {pre_drift:.4f}"
                    )
                    
                    if drift_impact < -5:
                        st.warning("Drift significantly impacted performance!")
                    else:
                        st.info("Drift impact is minimal.")
                else:
                    st.info("No drift was applied in this analysis.")
            
            # Feature importance (if available)
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importances = st.session_state.model.feature_importances_
                feature_names = st.session_state.X_train.columns
                
                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(15)
                
                # Plot
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 4: REPORTS
    # ========================================================================
    with tabs[3]:
        st.header("📄 Reports & Export")
        
        if not st.session_state.analysis_done:
            st.info("ℹ️ Run analysis first to generate reports")
        else:
            df_metrics = pd.DataFrame(st.session_state.metrics_history)
            problem_type = st.session_state.problem_type
            model_name = type(st.session_state.model).__name__
            
            st.subheader("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            # CSV Export
            with col1:
                st.markdown("### 📊 CSV Report")
                st.write("Download metrics as CSV file")
                
                csv = df_metrics.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"model_aging_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # HTML Export
            with col2:
                st.markdown("### 📄 HTML Report")
                st.write("Download comprehensive HTML report")
                
                html_report = generate_report_html(df_metrics, problem_type, model_name)
                st.download_button(
                    label="⬇️ Download HTML",
                    data=html_report,
                    file_name=f"model_aging_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            # Model Export
            with col3:
                st.markdown("### 🤖 Save Model")
                st.write("Download trained model")
                
                model_bytes = BytesIO()
                pickle.dump(st.session_state.model, model_bytes)
                model_bytes.seek(0)
                
                st.download_button(
                    label="⬇️ Download Model",
                    data=model_bytes,
                    file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Preview HTML report
            st.subheader("👀 Report Preview")
            
            with st.expander("View HTML Report Preview"):
                st.components.v1.html(html_report, height=800, scrolling=True)
            
            # Text report
            st.subheader("📝 Text Summary")
            
            main_metric = 'f1' if problem_type in ['binary', 'classification'] else 'r2'
            initial = df_metrics[main_metric].iloc[0]
            final = df_metrics[main_metric].iloc[-1]
            change = ((final - initial) / initial) * 100
            drift_count = df_metrics['drift_applied'].sum()
            
            report_text = f"""
MODEL AGING ANALYSIS REPORT
===========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_name}
Problem Type: {problem_type}

PERFORMANCE SUMMARY
-------------------
Initial {main_metric.upper()}: {initial:.4f}
Final {main_metric.upper()}: {final:.4f}
Change: {change:+.2f}%
Average {main_metric.upper()}: {df_metrics[main_metric].mean():.4f}

DRIFT ANALYSIS
--------------
Total Periods: {len(df_metrics)}
Drift Periods: {drift_count}
Drift Frequency: {drift_count/len(df_metrics)*100:.1f}%

RECOMMENDATIONS
---------------
"""
            
            if final < initial * 0.90:
                report_text += "\n⚠️ CRITICAL: Performance dropped >10%. Immediate retraining required!"
            elif final < initial * 0.95:
                report_text += "\n⚠️ WARNING: Performance dropped 5-10%. Consider adaptation."
            else:
                report_text += "\n✅ HEALTHY: Performance is acceptable. Continue monitoring."
            
            if drift_count > len(df_metrics) * 0.3:
                report_text += "\n⚠️ High drift frequency detected. Implement continuous learning."
            
            st.code(report_text, language=None)
            
            # Recommendations
            st.subheader("💡 Actionable Recommendations")

            recommendations = []

            if final < initial * 0.90:
                recommendations.append({
                    "priority": "🔴 HIGH",
                    "action": "Model Retraining",
                    "description": "Performance has degraded significantly. Retrain the model with recent data.",
                    "timeline": "Immediate"
                })

            if drift_count > len(df_metrics) * 0.3:
                recommendations.append({
                    "priority": "🟡 MEDIUM",
                    "action": "Implement Continuous Learning",
                    "description": "High drift frequency detected. Set up incremental learning pipeline.",
                    "timeline": "Within 1 week"
                })

            if not recommendations:
                recommendations.append({
                    "priority": "🟢 LOW",
                    "action": "Continue Monitoring",
                    "description": "Model performance is stable. Maintain current monitoring schedule.",
                    "timeline": "Ongoing"
                })

            for rec in recommendations:
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; 
                                margin-bottom: 15px; border-left: 4px solid #667eea;">
                        <h4 style="color: #ffffff; margin-top: 0;">{rec['priority']} - {rec['action']}</h4>
                        <p style="color: #e0e0e0; margin: 10px 0;"><strong>Description:</strong> {rec['description']}</p>
                        <p style="color: #e0e0e0; margin: 10px 0 0 0;"><strong>Timeline:</strong> {rec['timeline']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🎯 Model Aging Detection - İrem YÜCEL & Tolga ÇELİK</p>
            <p>Built with ❤️ using Streamlit | © 2025</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()