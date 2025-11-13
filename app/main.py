import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import load_models as lm

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Argumentos",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores mejorada - Máximo contraste y legibilidad
COLORS = {
    'primary': '#1E40AF',      # Azul oscuro vibrante
    'secondary': '#7C3AED',    # Púrpura
    'accent': '#F59E0B',       # Ámbar
    'success': '#059669',      # Verde oscuro
    'warning': '#EA580C',      # Naranja oscuro
    'danger': '#DC2626',       # Rojo oscuro
    'effective': '#059669',    # Verde oscuro para Effective
    'adequate': '#2563EB',     # Azul fuerte para Adequate
    'ineffective': '#EA580C',  # Naranja oscuro para Ineffective
    'background': '#FFFFFF',   # Blanco puro
    'text': '#0F172A',         # Negro azulado
    'text_secondary': '#1E293B', # Gris muy oscuro (en lugar de gris claro)
    'card_bg': '#F8FAFC'       # Gris muy claro para cards
}

# CSS personalizado
st.markdown(f"""
    <style>
    /* Fondo principal */
    .stApp {{
        background-color: {COLORS['background']};
    }}
    
    /* Títulos */
    h1 {{
        color: {COLORS['primary']};
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid {COLORS['accent']};
        margin-bottom: 1.5rem;
    }}
    
    h2 {{
        color: {COLORS['secondary']};
        font-weight: 600;
    }}
    
    h3 {{
        color: {COLORS['primary']};
        font-weight: 500;
    }}
    
    h4 {{
        color: {COLORS['text']};
        font-weight: 600;
    }}
    
    /* Todo el texto debe ser oscuro y visible */
    p, span, div, label {{
        color: {COLORS['text']} !important;
    }}
    
    /* Texto secundario con mejor contraste */
    .stMarkdown p {{
        color: {COLORS['text_secondary']} !important;
    }}
    
    /* Botones */
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white !important;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }}
    
    /* Selectbox y inputs con texto oscuro */
    .stSelectbox label, .stTextArea label {{
        color: {COLORS['text']} !important;
        font-weight: 600;
    }}
    
    /* Fix dropdown options visibility */
    .stSelectbox > div > div {{
        background-color: white !important;
        color: {COLORS['text']} !important;
    }}
    
    .stSelectbox [data-baseweb="select"] {{
        background-color: white !important;
    }}
    
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: white !important;
        color: {COLORS['text']} !important;
    }}
    
    /* Dropdown menu */
    [role="listbox"] {{
        background-color: white !important;
    }}
    
    [role="option"] {{
        background-color: white !important;
        color: {COLORS['text']} !important;
    }}
    
    [role="option"]:hover {{
        background-color: {COLORS['card_bg']} !important;
        color: {COLORS['primary']} !important;
    }}
    
    /* Métricas */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text']} !important;
        font-weight: 600;
    }}
    
    [data-testid="stMetricDelta"] {{
        color: {COLORS['success']} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #E5E7EB;
        border-right: 3px solid {COLORS['primary']};
    }}
    
    [data-testid="stSidebar"] * {{
        color: {COLORS['text']} !important;
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {COLORS['primary']} !important;
    }}
    
    /* Radio buttons en sidebar */
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        color: {COLORS['text']} !important;
        font-weight: 600;
        font-size: 1rem;
    }}
    
    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] {{
        background-color: {COLORS['primary']} !important;
    }}
    
    [data-testid="stSidebar"] [role="radiogroup"] [data-checked="true"] * {{
        color: white !important;
    }}
    
    /* Cards personalizados */
    .metric-card {{
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid {COLORS['primary']};
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }}
    
    .metric-card p {{
        color: {COLORS['text']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: {COLORS['card_bg']};
        padding: 0.5rem;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 1.1rem;
        color: {COLORS['text']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        border-radius: 8px;
    }}
    
    /* Info boxes con texto oscuro */
    .info-box {{
        background: linear-gradient(135deg, {COLORS['primary']}10 0%, {COLORS['accent']}10 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {COLORS['accent']};
        margin: 1rem 0;
    }}
    
    .info-box * {{
        color: {COLORS['text']} !important;
    }}
    
    /* Radio buttons */
    [data-testid="stRadio"] label {{
        font-weight: 600;
        color: {COLORS['text']} !important;
    }}
    
    /* Dataframe */
    .stDataFrame {{
        color: {COLORS['text']} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        color: {COLORS['text']} !important;
        font-weight: 600;
    }}
    
    /* Success/Info/Warning/Error boxes */
    .stAlert {{
        color: {COLORS['text']} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Device setup
device = "cpu"

def evaluate_models_safe(device, distilbert_model, deberta_model, svm_model, tfidf_vectorizer, use_train=False):
    """Versión segura de evaluate_models que maneja diferentes nombres de columnas"""
    
    if use_train:
        # Usar train set con split
        from sklearn.model_selection import train_test_split
        
        full_dataset = pd.read_csv("../data/clean_train.csv")
        
        # Detectar columna de texto
        if 'discourse_text_clean' in full_dataset.columns:
            text_col = 'discourse_text_clean'
        elif 'discourse_text' in full_dataset.columns:
            text_col = 'discourse_text'
        else:
            text_cols = [col for col in full_dataset.columns if 'text' in col.lower() and 'essay' not in col.lower()]
            text_col = text_cols[0] if text_cols else None
        
        if not text_col or 'label' not in full_dataset.columns:
            st.error("No se pudo preparar el dataset para evaluación")
            return None
        
        # Split 80/20
        _, test_subset = train_test_split(
            full_dataset, 
            test_size=0.2, 
            random_state=42, 
            stratify=full_dataset['label']
        )
        
        X_test = test_subset[text_col]
        y_true = test_subset['label']
        
        st.info(f"Evaluando con {len(test_subset)} muestras del train set (20% para validación)")
        
    else:
        # Usar test set normal
        test_dataset = pd.read_csv("../data/clean_test.csv")
        
        # Detectar columna de texto
        if 'discourse_text_clean' in test_dataset.columns:
            text_col = 'discourse_text_clean'
        elif 'discourse_text' in test_dataset.columns:
            text_col = 'discourse_text'
        else:
            text_cols = [col for col in test_dataset.columns if 'text' in col.lower() and 'essay' not in col.lower()]
            if text_cols:
                text_col = text_cols[0]
            else:
                st.error("No se encontró columna de texto en test dataset")
                return None
        
        # Verificar que exista la columna de labels
        if 'label' not in test_dataset.columns:
            st.error("El dataset de prueba no tiene la columna 'label'. No se puede evaluar el rendimiento.")
            return None
        
        X_test = test_dataset[text_col]
        y_true = test_dataset["label"]
    
    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    
    # Evaluate DistilBERT
    st.write("Evaluando DistilBERT...")
    y_pred = [lm.predict_distilbert(text, device, distilbert_model) for text in X_test]
    results["Model"].append("DistilBERT")
    results["Accuracy"].append(accuracy_score(y_true, y_pred))
    results["Precision"].append(precision_score(y_true, y_pred, average="macro"))
    results["Recall"].append(recall_score(y_true, y_pred, average="macro"))
    results["F1-Score"].append(f1_score(y_true, y_pred, average="macro"))
    
    # Evaluate DeBERTa
    st.write("Evaluando DeBERTa...")
    y_pred = [lm.predict_deberta(text, device, deberta_model) for text in X_test]
    results["Model"].append("DeBERTa")
    results["Accuracy"].append(accuracy_score(y_true, y_pred))
    results["Precision"].append(precision_score(y_true, y_pred, average="macro"))
    results["Recall"].append(recall_score(y_true, y_pred, average="macro"))
    results["F1-Score"].append(f1_score(y_true, y_pred, average="macro"))
    
    # Evaluate SVM
    st.write("Evaluando SVM + TF-IDF...")
    y_pred = [lm.predict_svm(text, svm_model, tfidf_vectorizer) for text in X_test]
    label_map = {0: 'Ineffective', 1: 'Adequate', 2: 'Effective'}
    y_pred_mapped = [label_map[p] if isinstance(p, int) else p for p in y_pred]
    y_true_mapped = [label_map[y] if isinstance(y, int) else y for y in y_true]
    results["Model"].append("SVM + TF-IDF")
    results["Accuracy"].append(accuracy_score(y_true_mapped, y_pred_mapped))
    results["Precision"].append(precision_score(y_true_mapped, y_pred_mapped, average="macro"))
    results["Recall"].append(recall_score(y_true_mapped, y_pred_mapped, average="macro"))
    results["F1-Score"].append(f1_score(y_true_mapped, y_pred_mapped, average="macro"))
    
    df = pd.DataFrame(results)
    return df

# Cargar modelos en caché
@st.cache_resource
def load_all_models():
    """Carga todos los modelos una sola vez"""
    with st.spinner(" Cargando modelos..."):
        distilbert = lm.load_distilbert(device)
        deberta = lm.load_deberta(device)
        svm, tfidf = lm.load_svm()
    return distilbert, deberta, svm, tfidf

# Cargar datos en caché
@st.cache_data
def load_datasets():
    """Carga los datasets de entrenamiento y prueba"""
    try:
        train_df = pd.read_csv("../data/clean_train.csv")
        test_df = pd.read_csv("../data/clean_test.csv")
    except FileNotFoundError:
        st.error(" No se encontraron los archivos de datos. Verifica que estén en ../data/")
        st.stop()
    
    # Identificar columna de texto en TRAIN
    if 'discourse_text_clean' in train_df.columns:
        train_text_col = 'discourse_text_clean'
    elif 'discourse_text' in train_df.columns:
        train_text_col = 'discourse_text'
    else:
        train_text_cols = [col for col in train_df.columns if 'text' in col.lower() and 'essay' not in col.lower()]
        if train_text_cols:
            train_text_col = train_text_cols[0]
        else:
            st.error(f" No se encontró columna de texto en train. Columnas: {train_df.columns.tolist()}")
            st.stop()
    
    # Identificar columna de texto en TEST
    if 'discourse_text_clean' in test_df.columns:
        test_text_col = 'discourse_text_clean'
    elif 'discourse_text' in test_df.columns:
        test_text_col = 'discourse_text'
    else:
        test_text_cols = [col for col in test_df.columns if 'text' in col.lower() and 'essay' not in col.lower()]
        if test_text_cols:
            test_text_col = test_text_cols[0]
        else:
            test_text_col = None  # Test puede no tener texto
    
    st.sidebar.info(f" Train: `{train_text_col}`\n\n Test: `{test_text_col if test_text_col else 'N/A'}`")
    
    # Mapeo de labels
    label_map = {0: 'Ineffective', 1: 'Adequate', 2: 'Effective'}
    
    if 'label' in train_df.columns:
        train_df['label_name'] = train_df['label'].map(label_map)
    else:
        st.error(" No se encontró la columna 'label' en el dataset de entrenamiento")
        st.stop()
    
    # Solo mapear label_name si existe la columna 'label' en test
    if 'label' in test_df.columns:
        test_df['label_name'] = test_df['label'].map(label_map)
    
    # Calcular longitud de textos SOLO si existen las columnas
    if train_text_col in train_df.columns:
        train_df['text_length'] = train_df[train_text_col].str.len()
        train_df['word_count'] = train_df[train_text_col].str.split().str.len()
    
    if test_text_col and test_text_col in test_df.columns:
        test_df['text_length'] = test_df[test_text_col].str.len()
        test_df['word_count'] = test_df[test_text_col].str.split().str.len()
    
    return train_df, test_df, train_text_col

# Sidebar - Navegación
with st.sidebar:
    st.title("Clasificador de Argumentos")
    st.markdown("---")
    
    # Selector de página
    page = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Exploración de Datos", "🤖 Clasificador", "📈 Rendimiento de Modelos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Información del modelo
    with st.expander("Información"):
        st.markdown("""
        **Modelos disponibles:**
        - DistilBERT
        - DeBERTa
        - SVM + TF-IDF
        
        **Clases:**
        - 🟢 Effective
        - 🔵 Adequate
        - 🟠 Ineffective
        """)

# Cargar modelos y datos
distilbert_model, deberta_model, svm_model, tfidf_vectorizer = load_all_models()
train_df, test_df, text_col = load_datasets()

st.sidebar.success("Modelos cargados correctamente")

# ==================== PÁGINA: INICIO ====================
if page == "🏠 Inicio":
    st.title("Bienvenido al Clasificador de Argumentos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {COLORS['primary']};">📚 Dataset</h3>
            <p style="font-size: 2rem; font-weight: 700; color: {COLORS['accent']};">
                {len(train_df) + len(test_df)}
            </p>
            <p style="color: {COLORS['text_secondary']}; font-weight: 600;">argumentos totales</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {COLORS['primary']};">🤖 Modelos</h3>
            <p style="font-size: 2rem; font-weight: 700; color: {COLORS['accent']};">3</p>
            <p style="color: {COLORS['text_secondary']}; font-weight: 600;">algoritmos disponibles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {COLORS['primary']};">🎯 Clases</h3>
            <p style="font-size: 2rem; font-weight: 700; color: {COLORS['accent']};">3</p>
            <p style="color: {COLORS['text_secondary']}; font-weight: 600;">categorías de clasificación</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3> Sobre el Proyecto</h3>
        <p>
        Esta aplicación utiliza técnicas de <strong>Procesamiento de Lenguaje Natural (NLP)</strong> 
        para clasificar argumentos en tres categorías: <strong>Effective</strong>, <strong>Adequate</strong>, 
        y <strong>Ineffective</strong>.
        </p>
        <p>
        Puedes explorar los datos, clasificar nuevos argumentos en tiempo real, y comparar el 
        rendimiento de diferentes modelos de Machine Learning y Deep Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###  Funcionalidades")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ** Exploración de Datos**
        - Visualización de distribución de clases
        - Análisis de longitud de textos
        - Word clouds interactivos
        - Estadísticas descriptivas
        """)
    
    with col2:
        st.markdown("""
        ** Clasificación Inteligente**
        - Clasificación con 3 modelos diferentes
        - Comparación de predicciones
        - Análisis de rendimiento
        - Métricas detalladas
        """)

# ==================== PÁGINA: EXPLORACIÓN DE DATOS ====================
elif page == "📊 Exploración de Datos":
    st.title(" Exploración de Datos")
    st.markdown("Analiza las características y distribución del conjunto de datos de entrenamiento")
    
    # Tabs para organizar las visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribución", "📏 Análisis de Texto", "☁️ Word Clouds", "🔢 Estadísticas"])
    
    with tab1:
        st.subheader("Distribución de Clases")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de barras interactivo
            class_counts = train_df['label_name'].value_counts().reset_index()
            class_counts.columns = ['Clase', 'Cantidad']
            
            fig = px.bar(
                class_counts,
                x='Clase',
                y='Cantidad',
                color='Clase',
                color_discrete_map={
                    'Effective': COLORS['effective'],
                    'Adequate': COLORS['adequate'],
                    'Ineffective': COLORS['ineffective']
                },
                title="Distribución de Clases en el Dataset de Entrenamiento",
                text='Cantidad'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Clase",
                yaxis_title="Número de Argumentos"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("###  Resumen")
            for label, count in class_counts.itertuples(index=False):
                percentage = (count / len(train_df)) * 100
                st.metric(
                    label=label,
                    value=f"{count}",
                    delta=f"{percentage:.1f}%"
                )
        
        # Gráfico de pie
        fig_pie = px.pie(
            class_counts,
            values='Cantidad',
            names='Clase',
            title="Proporción de Clases",
            color='Clase',
            color_discrete_map={
                'Effective': COLORS['effective'],
                'Adequate': COLORS['adequate'],
                'Ineffective': COLORS['ineffective']
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Distribución por tipo de discurso
        st.subheader("Distribución por Tipo de Discurso")
        discourse_dist = train_df.groupby(['discourse_type', 'label_name']).size().reset_index(name='count')
        
        fig_discourse = px.bar(
            discourse_dist,
            x='discourse_type',
            y='count',
            color='label_name',
            barmode='group',
            title="Clases por Tipo de Discurso",
            color_discrete_map={
                'Effective': COLORS['effective'],
                'Adequate': COLORS['adequate'],
                'Ineffective': COLORS['ineffective']
            },
            labels={'discourse_type': 'Tipo de Discurso', 'count': 'Cantidad', 'label_name': 'Clase'}
        )
        fig_discourse.update_layout(height=500)
        st.plotly_chart(fig_discourse, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis de Longitud de Textos")
        
        # Selector de métrica
        metric_choice = st.radio(
            "Selecciona la métrica a analizar:",
            ["Longitud de Caracteres", "Número de Palabras"],
            horizontal=True
        )
        
        metric_col = 'text_length' if metric_choice == "Longitud de Caracteres" else 'word_count'
        metric_label = 'Caracteres' if metric_choice == "Longitud de Caracteres" else 'Palabras'
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig_box = px.box(
                train_df,
                x='label_name',
                y=metric_col,
                color='label_name',
                title=f"Distribución de {metric_label} por Clase",
                color_discrete_map={
                    'Effective': COLORS['effective'],
                    'Adequate': COLORS['adequate'],
                    'Ineffective': COLORS['ineffective']
                },
                labels={'label_name': 'Clase', metric_col: f'Número de {metric_label}'}
            )
            fig_box.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Violin plot
            fig_violin = px.violin(
                train_df,
                x='label_name',
                y=metric_col,
                color='label_name',
                title=f"Densidad de {metric_label} por Clase",
                color_discrete_map={
                    'Effective': COLORS['effective'],
                    'Adequate': COLORS['adequate'],
                    'Ineffective': COLORS['ineffective']
                },
                box=True,
                labels={'label_name': 'Clase', metric_col: f'Número de {metric_label}'}
            )
            fig_violin.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # Histograma interactivo
        fig_hist = px.histogram(
            train_df,
            x=metric_col,
            color='label_name',
            marginal='box',
            title=f"Distribución de {metric_label} en todos los Argumentos",
            color_discrete_map={
                'Effective': COLORS['effective'],
                'Adequate': COLORS['adequate'],
                'Ineffective': COLORS['ineffective']
            },
            labels={metric_col: f'Número de {metric_label}', 'label_name': 'Clase'},
            nbins=50
        )
        fig_hist.update_layout(height=500)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Estadísticas por clase
        st.subheader(f" Estadísticas de {metric_label} por Clase")
        stats_df = train_df.groupby('label_name')[metric_col].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.subheader("☁️ Nubes de Palabras por Clase")
        
        class_selector = st.selectbox(
            "Selecciona una clase para visualizar:",
            ['Effective', 'Adequate', 'Ineffective']
        )
        
        # Filtrar textos por clase usando la columna correcta
        class_texts = train_df[train_df['label_name'] == class_selector][text_col]
        all_text = ' '.join(class_texts.astype(str))
        
        # Generar word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        # Mostrar word cloud
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Palabras más frecuentes en argumentos {class_selector}', 
                    fontsize=20, fontweight='bold', pad=20)
        st.pyplot(fig)
        
        # Mostrar top palabras
        st.subheader(f" Top 20 Palabras en {class_selector}")
        
        from collections import Counter
        words = all_text.lower().split()
        word_freq = Counter(words).most_common(20)
        word_df = pd.DataFrame(word_freq, columns=['Palabra', 'Frecuencia'])
        
        fig_words = px.bar(
            word_df,
            x='Frecuencia',
            y='Palabra',
            orientation='h',
            title=f'Top 20 Palabras más Frecuentes - {class_selector}',
            color='Frecuencia',
            color_continuous_scale='viridis'
        )
        fig_words.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_words, use_container_width=True)
    
    with tab4:
        st.subheader("🔢 Estadísticas Descriptivas del Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Dataset de Entrenamiento")
            st.metric("Total de argumentos", len(train_df))
            st.metric("Tipos de discurso únicos", train_df['discourse_type'].nunique())
            st.metric("Promedio de caracteres", f"{train_df['text_length'].mean():.0f}")
            st.metric("Promedio de palabras", f"{train_df['word_count'].mean():.0f}")
        
        with col2:
            st.markdown("###  Dataset de Prueba")
            st.metric("Total de argumentos", len(test_df))
            st.metric("Tipos de discurso únicos", test_df['discourse_type'].nunique() if 'discourse_type' in test_df.columns else "N/A")
            
            # Solo mostrar métricas de texto si existen
            if 'text_length' in test_df.columns:
                st.metric("Promedio de caracteres", f"{test_df['text_length'].mean():.0f}")
            if 'word_count' in test_df.columns:
                st.metric("Promedio de palabras", f"{test_df['word_count'].mean():.0f}")
        
        st.markdown("---")
        
        # Tabla de estadísticas completa
        st.subheader(" Estadísticas Detalladas - Training Set")
        
        stats_summary = pd.DataFrame({
            'Métrica': ['Total', 'Effective', 'Adequate', 'Ineffective'],
            'Cantidad Train': [
                len(train_df),
                len(train_df[train_df['label_name'] == 'Effective']),
                len(train_df[train_df['label_name'] == 'Adequate']),
                len(train_df[train_df['label_name'] == 'Ineffective'])
            ],
            '% del Total': [
                100.0,
                (len(train_df[train_df['label_name'] == 'Effective']) / len(train_df)) * 100,
                (len(train_df[train_df['label_name'] == 'Adequate']) / len(train_df)) * 100,
                (len(train_df[train_df['label_name'] == 'Ineffective']) / len(train_df)) * 100
            ]
        })
        
        # Formatear porcentajes
        stats_summary['% del Total'] = stats_summary['% del Total'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(stats_summary, use_container_width=True, hide_index=True)
        
        # Análisis de texto solo con train
        st.subheader(" Análisis de Longitud de Texto (Training)")
        
        text_stats = train_df.groupby('label_name')[['text_length', 'word_count']].describe().round(2)
        st.dataframe(text_stats, use_container_width=True)

# ==================== PÁGINA: CLASIFICADOR ====================
elif page == "🤖 Clasificador":
    st.title(" Clasificador de Argumentos")
    st.markdown("Clasifica nuevos argumentos usando los modelos entrenados")
    
    # Selector de modelo
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_choice = st.selectbox(
            "🔧 Selecciona el modelo a utilizar:",
            ("DistilBERT", "DeBERTa", "SVM + TF-IDF", "Todos los Modelos"),
            help="Elige un modelo específico o prueba con todos para comparar"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_info = st.checkbox(" Mostrar info", value=False)
    
    if show_info:
        st.info("""
        **DistilBERT**: Modelo transformer eficiente, balance entre velocidad y precisión.  
        **DeBERTa**: Modelo transformer avanzado con mejor comprensión contextual.  
        **SVM + TF-IDF**: Modelo clásico de ML, rápido y efectivo para textos.
        """)
    
    st.markdown("---")
    
    # Área de texto para el argumento
    argument = st.text_area(
        label=" Escribe tu argumento aquí:",
        placeholder="Ejemplo: Students should be required to study abroad because it broadens their cultural perspectives and helps them develop independence...",
        key="argument_text",
        height=200,
        help="Escribe el argumento que deseas clasificar"
    )
    
    # Botón de clasificación
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_btn = st.button(
            "🚀 Clasificar Argumento",
            key="classify_btn",
            use_container_width=True,
            type="primary"
        )
    
    if classify_btn:
        if argument.strip() == "":
            st.warning(" Por favor escribe un argumento antes de clasificar.")
        else:
            with st.spinner(" Clasificando tu argumento..."):
                results = {}
                
                # Realizar predicciones según el modelo seleccionado
                if model_choice in ["DistilBERT", "Todos los Modelos"]:
                    pred = lm.predict_distilbert(argument, device, distilbert_model)
                    label_map = {0: 'Ineffective', 1: 'Adequate', 2: 'Effective'}
                    results["DistilBERT"] = label_map[pred]
                
                if model_choice in ["DeBERTa", "Todos los Modelos"]:
                    pred = lm.predict_deberta(argument, device, deberta_model)
                    results["DeBERTa"] = pred
                
                if model_choice in ["SVM + TF-IDF", "Todos los Modelos"]:
                    pred = lm.predict_svm(argument, svm_model, tfidf_vectorizer)
                    label_map = {0: 'Ineffective', 1: 'Adequate', 2: 'Effective'}
                    results["SVM + TF-IDF"] = label_map[pred]
                
                # Guardar en session state
                st.session_state["last_prediction"] = results
                st.session_state["last_argument"] = argument
            
            st.success(" Clasificación completada!")
            
            st.markdown("---")
            st.subheader(" Resultados de la Clasificación")
            
            # Mostrar resultados en columnas
            if len(results) == 1:
                # Un solo modelo
                model_name, prediction = list(results.items())[0]
                
                # Color según la predicción
                if prediction == 'Effective':
                    color = COLORS['effective']
                    icon = "🟢"
                elif prediction == 'Adequate':
                    color = COLORS['adequate']
                    icon = "🔵"
                else:
                    color = COLORS['ineffective']
                    icon = "🟠"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                            padding: 2rem; border-radius: 12px; border-left: 5px solid {color}; text-align: center;">
                    <h2 style="color: {color}; margin: 0;">{icon} {prediction}</h2>
                    <p style="color: {COLORS['text_secondary']}; margin-top: 0.5rem; font-weight: 600;">Predicción de {model_name}</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                # Múltiples modelos
                cols = st.columns(len(results))
                
                for idx, (model_name, prediction) in enumerate(results.items()):
                    with cols[idx]:
                        # Color según la predicción
                        if prediction == 'Effective':
                            color = COLORS['effective']
                            icon = "🟢"
                        elif prediction == 'Adequate':
                            color = COLORS['adequate']
                            icon = "🔵"
                        else:
                            color = COLORS['ineffective']
                            icon = "🟠"
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid {color}; 
                                    text-align: center; height: 180px; display: flex; 
                                    flex-direction: column; justify-content: center;">
                            <h4 style="color: {COLORS['text']}; margin: 0;">{model_name}</h4>
                            <h2 style="color: {color}; margin: 0.5rem 0;">{icon}</h2>
                            <h3 style="color: {color}; margin: 0;">{prediction}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Consenso
                st.markdown("---")
                predictions_list = list(results.values())
                
                if len(set(predictions_list)) == 1:
                    st.success(f" **Consenso total**: Todos los modelos coinciden en que el argumento es **{predictions_list[0]}**")
                else:
                    from collections import Counter
                    most_common = Counter(predictions_list).most_common(1)[0]
                    st.info(f" **Predicción mayoritaria**: **{most_common[0]}** ({most_common[1]}/{len(predictions_list)} modelos)")
    
    # Mostrar últimas predicciones
    if "last_prediction" in st.session_state:
        st.markdown("---")
        st.subheader(" Última Clasificación")
        
        with st.expander("Ver argumento clasificado"):
            st.markdown(f"**Texto:** {st.session_state['last_argument']}")
            st.markdown("**Predicciones:**")
            for model_name, pred in st.session_state["last_prediction"].items():
                st.markdown(f"- **{model_name}**: `{pred}`")

# ==================== PÁGINA: RENDIMIENTO ====================
elif page == "📈 Rendimiento de Modelos":
    st.title(" Rendimiento de Modelos")
    st.markdown("Compara el desempeño de los modelos en el conjunto de datos de prueba")
    
    # Verificar si el dataset de prueba tiene labels
    has_labels = 'label' in test_df.columns
    
    if not has_labels:
        st.info("""
         **El dataset de prueba no contiene labels**
        
        Para evaluar el rendimiento, usaremos una **validación cruzada** sobre el dataset de entrenamiento.
        Esto es una práctica estándar en Machine Learning cuando no tienes un test set con labels.
        
        Se dividirá el training set en: 80% entrenamiento / 20% validación
        """)
        
        # Usar el train set para evaluación
        eval_df = train_df.copy()
        use_train_for_eval = True
    else:
        st.success("✅ El dataset de prueba tiene labels. Se usará para la evaluación.")
        eval_df = test_df.copy()
        use_train_for_eval = False
    
    # Botón para calcular rendimiento
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calc_performance = st.button(
            " Calcular Rendimiento en Test Set",
            use_container_width=True,
            type="primary"
        )
    
    if calc_performance or "performance_data" in st.session_state:
        
        if calc_performance:
            with st.spinner(" Evaluando modelos en el conjunto de prueba... (esto puede tomar un minuto)"):
                performance_data = evaluate_models_safe(
                    device, 
                    distilbert_model, 
                    deberta_model, 
                    svm_model, 
                    tfidf_vectorizer,
                    use_train=use_train_for_eval
                )
                
                if performance_data is None:
                    st.error("No se pudo evaluar. Verifica los datos.")
                    st.stop()
                
                st.session_state["performance_data"] = performance_data
        else:
            performance_data = st.session_state["performance_data"]
        
        st.success(" Evaluación completada!")
        
        # Tabs para diferentes visualizaciones
        tab1, tab2, tab3 = st.tabs(["📊 Comparación General", "🎯 Métricas Detalladas", "📋 Tabla de Resultados"])
        
        with tab1:
            st.subheader("Comparación de Métricas")
            
            # Gráfico de barras agrupadas
            fig = px.bar(
                performance_data.melt(id_vars=["Model"], var_name="Métrica", value_name="Score"),
                x="Model",
                y="Score",
                color="Métrica",
                barmode="group",
                title="Comparación de Rendimiento entre Modelos",
                text_auto='.3f',
                color_discrete_map={
                    'Accuracy': COLORS['primary'],
                    'Precision': COLORS['secondary'],
                    'Recall': COLORS['accent'],
                    'F1-Score': COLORS['success']
                }
            )
            fig.update_layout(
                height=500,
                yaxis_title="Score",
                xaxis_title="Modelo",
                legend_title="Métrica"
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de radar
            st.subheader("Análisis Multidimensional")
            
            fig_radar = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for idx, row in performance_data.iterrows():
                model_name = row['Model']
                values = [row[metric] for metric in metrics]
                values.append(values[0])  # Cerrar el polígono
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model_name
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Perfil de Rendimiento por Modelo",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab2:
            st.subheader("Métricas Individuales por Modelo")
            
            for idx, row in performance_data.iterrows():
                with st.expander(f" {row['Model']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Accuracy",
                            f"{row['Accuracy']:.4f}",
                            delta=f"{(row['Accuracy'] - performance_data['Accuracy'].mean()):.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Precision",
                            f"{row['Precision']:.4f}",
                            delta=f"{(row['Precision'] - performance_data['Precision'].mean()):.4f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Recall",
                            f"{row['Recall']:.4f}",
                            delta=f"{(row['Recall'] - performance_data['Recall'].mean()):.4f}"
                        )
                    
                    with col4:
                        st.metric(
                            "F1-Score",
                            f"{row['F1-Score']:.4f}",
                            delta=f"{(row['F1-Score'] - performance_data['F1-Score'].mean()):.4f}"
                        )
        
        with tab3:
            st.subheader("Tabla Comparativa de Resultados")
            
            # Formatear DataFrame
            styled_df = performance_data.copy()
            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Identificar el mejor modelo
            best_model_idx = performance_data['F1-Score'].idxmax()
            best_model = performance_data.loc[best_model_idx, 'Model']
            best_f1 = performance_data.loc[best_model_idx, 'F1-Score']
            
            st.success(f" **Mejor modelo según F1-Score**: **{best_model}** ({best_f1:.4f})")
            
            # Análisis adicional
            st.markdown("---")
            st.subheader(" Análisis Estadístico")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Promedios por Métrica**")
                avg_stats = performance_data[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
                for metric, value in avg_stats.items():
                    st.markdown(f"- {metric}: `{value:.4f}`")
            
            with col2:
                st.markdown("**Desviación Estándar**")
                std_stats = performance_data[['Accuracy', 'Precision', 'Recall', 'F1-Score']].std()
                for metric, value in std_stats.items():
                    st.markdown(f"- {metric}: `{value:.4f}`")
    
    else:
        st.info("👆 Haz clic en el botón para calcular el rendimiento de los modelos en el conjunto de prueba.")
        
        st.markdown("""
        <div class="info-box">
            <h4> Sobre las Métricas</h4>
            <ul>
                <li><strong>Accuracy</strong>: Proporción de predicciones correctas sobre el total.</li>
                <li><strong>Precision</strong>: De las predicciones positivas, cuántas fueron correctas.</li>
                <li><strong>Recall</strong>: De los casos positivos reales, cuántos se detectaron.</li>
                <li><strong>F1-Score</strong>: Media armónica entre Precision y Recall.</li>
            </ul>
            <p>Todas las métricas están calculadas con promedio <code>macro</code>, 
            lo que significa que se calcula la métrica para cada clase y se promedian los resultados.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['text_secondary']}; padding: 2rem 0;">
    <p style="font-weight: 600;"> Proyecto de Clasificación de Argumentos</p>
    <p style="font-size: 0.9rem;">Desarrollado con Streamlit, PyTorch, y Transformers</p>
</div>
""", unsafe_allow_html=True)