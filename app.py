import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP PCA - Master IAENG", layout="wide")
st.header("EL MEHDI - Master IAENG")
st.title("TP PCA : Compression, Reconstruction & Débruitage sur MNIST")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    # Filtrage pour ne garder que les 0 et les 1 (par cohérence avec le TP LDA)
    mask = (y == 0) | (y == 1)
    return X[mask][:3000], y[mask][:3000]

with st.spinner("Chargement et préparation des données MNIST..."):
    X, y = load_mnist()

# --- INTERFACE CORPS ---
col_ctrl, col_visu = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Paramètres de l'ACP")
    
    # Choisir une image à tester dans le dataset
    max_idx = len(X) - 1
    img_idx = st.slider("Sélectionner une image du dataset", 0, max_idx, 0)
    image_originale = X[img_idx]
    
    st.markdown("---")
    st.write("### 🎛️ Mode de sélection des composantes")
    mode = st.radio("Choisir le type de test :", [
        "Test 1 : PCA-1 (Retour en arrière extrême)", 
        "Test 2 & 3 : Variance ciblée (Optimisation)", 
        "Test 4 : Noise Cancellation (Débruitage)"
    ])

    # Logique selon le test choisi
    if mode == "Test 1 : PCA-1 (Retour en arrière extrême)":
        n_components = 1
        st.info("PCA-1 : L'image va être écrasée sur un seul axe discriminant avant le 'retour en arrière'.")
        
    elif mode == "Test 2 & 3 : Variance ciblée (Optimisation)":
        variance_target = st.slider("Variance expliquée ciblée", 0.10, 0.99, 0.40, step=0.05)
        st.write(f"Cible actuelle : **{variance_target*100:.0f}%** de l'information.")
        
        # Fonction mise en cache pour accélérer la recherche de la variance
        @st.cache_data
        def get_components_from_variance(X_data, target):
            pca_temp = PCA().fit(X_data)
            cum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            return int(np.argmax(cum_variance >= target) + 1)
            
        n_components = get_components_from_variance(X, variance_target)
        st.success(f"Nombre d'axes optimisés à retenir : **{n_components}**")

    else: # Test 4 : Noise Cancellation
        n_components = st.slider("Nombre d'axes à conserver (Filtrage du bruit)", 5, 100, 30)
        # Ajout de bruit blanc artificiel sur l'image sélectionnée
        bruit = np.random.normal(0, 0.3, image_originale.shape)
        image_originale = np.clip(image_originale + bruit, 0, 1)
        st.info("Un bruit blanc a été injecté dans l'image d'origine. Les derniers axes de la PCA vont tenter de le filtrer.")

# --- 2. LOGIQUE MATHÉMATIQUE DE LA PCA (ALLER & RETOUR EN ARRIÈRE) ---
@st.cache_resource
def compute_pca(X_data, n_comp):
    pca_model = PCA(n_components=n_comp)
    pca_model.fit(X_data)
    return pca_model

with st.spinner("Calcul de la PCA en cours..."):
    pca = compute_pca(X, n_components)

# Encodage et reconstruction spécifique de l'image choisie
img_compressed = pca.transform(image_originale.reshape(1, -1))
image_reconstruite = pca.inverse_transform(img_compressed).reshape(28, 28)

# --- 3. RENDU ET VISUALISATION STREAMLIT ---
with col_visu:
    st.header("📊 Visualisation des Résultats")
    
    # Affichage des images Côte à Côte
    fig_imgs, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    title_orig = "Image avec bruit blanc" if "Noise" in mode else "Image Originale (784 px)"
    ax1.imshow(image_originale.reshape(28, 28), cmap='gray')
    ax1.set_title(title_orig)
    ax1.axis('off')
    
    ax2.imshow(image_reconstruite, cmap='gray')
    ax2.set_title(f"Reconstruction (PCA-{n_components})")
    ax2.axis('off')
    
    st.pyplot(fig_imgs)
    
    st.markdown("---")
    st.write("### 🗜️ L'image sous sa forme COMPRESSÉE (Ce que stocke la machine)")
    
    fig_comp, ax_comp = plt.subplots(figsize=(8, 1.2))
    im = ax_comp.imshow(img_compressed, cmap='viridis', aspect='auto')
    ax_comp.set_yticks([])  # Pas d'axe vertical, c'est un vecteur 1D
    ax_comp.set_xlabel("Coordonnées dans le nouvel espace des composantes principales")
    ax_comp.set_title(f"Vecteur compressé : {img_compressed.shape[1]} valeur(s) au lieu de 784 pixels !")
    
    fig_comp.colorbar(im, ax=ax_comp, orientation='horizontal', pad=0.5)
    st.pyplot(fig_comp)
    
    if n_components <= 10:
        st.write("**Valeurs numériques exactes envoyées pour la reconstruction :**", img_compressed[0])
        
    # Graphique de la variance cumulée (Scree Plot)
    st.markdown("---")
    st.write("### 📈 Courbe de l'optimisation de la variance (Scree Plot)")
    
    @st.cache_data
    def get_full_pca_variance(X_data):
        pca_full = PCA(n_components=min(100, len(X_data))).fit(X_data)
        return np.cumsum(pca_full.explained_variance_ratio_)
        
    cum_var_full = get_full_pca_variance(X)
    
    fig_var, ax_var = plt.subplots(figsize=(10, 3.5))
    ax_var.plot(range(1, len(cum_var_full) + 1), cum_var_full, marker='o', linestyle='-', color='#3498db', markersize=4)
    ax_var.axvline(x=n_components, color='red', linestyle='--', label=f'Axes conservés ({n_components})')
    ax_var.set_xlabel("Nombre de composantes (Axes)")
    ax_var.set_ylabel("Variance expliquée cumulée")
    ax_var.set_title("Recherche du 'Coude' d'optimisation")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend()
    
    st.pyplot(fig_var)
