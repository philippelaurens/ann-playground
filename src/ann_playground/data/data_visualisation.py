import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_learning_curves(history_df, model_name="MLP"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Courbe de la fonction de perte (Loss)
    ax1.plot(history_df['loss'], label='Entraînement', linewidth=2)
    ax1.plot(history_df['val_loss'], label='Validation', linewidth=2, linestyle='--')
    ax1.set_title(f'Évolution de la Perte ({model_name})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Courbe de la précision (Accuracy)
    ax2.plot(history_df['accuracy'], label='Entraînement', linewidth=2)
    ax2.plot(history_df['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
    ax2.set_title(f'Évolution de la Précision ({model_name})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig


def plot_pca_cumulative_variance(X_train_prep, threshold=0.95, random_state=42):
    """
    Ajuste une PCA sans limite, calcule le nombre de composantes pour atteindre 
    le seuil de variance désiré, et retourne la figure Matplotlib ainsi que le nombre de composantes.
    """
    # 1. Ajuster une PCA sans limite sur les données d'entraînement prétraitées
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_train_prep)

    # 2. Calculer la variance cumulée expliquée
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 3. Trouver automatiquement le nombre de composantes pour atteindre le seuil
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    # 4. Générer la courbe de la variance cumulée
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    ax.axhline(y=threshold, color='r', linestyle='-', label=f'{int(threshold*100)}% de variance')
    ax.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} composantes')

    ax.set_xlabel('Nombre de composantes principales')
    ax.set_ylabel('Variance cumulée expliquée')
    ax.set_title('Évolution de la variance cumulée expliquée')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()

    # On retourne à la fois la figure pour l'affichage et le nombre calculé si besoin
    return fig, n_components


def scree_plot(X_train_prep, figsize=(10, 6), random_state=42):
    """
    Calcule la PCA, génère l'éboulis des valeurs propres et retourne 
    la figure, l'objet PCA et les variances.
    """
    pca = PCA(random_state=random_state)
    pca.fit(X_train_prep)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(1, len(explained_variance) + 1),
           explained_variance, 
           alpha=0.5, align='center',
           label='Variance expliquée par composante', 
           color='#439cc8')
    
    ax.plot(range(1, len(cumulative_variance) + 1),
            cumulative_variance, 
            marker='o', linestyle='--',
            color='darkorange', 
            label='Variance expliquée cumulative')
            
    ax.set_xlabel('Composantes principales')
    ax.set_ylabel('Variance expliquée')
    ax.set_title('Éboulis des valeurs propres\n')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # On retourne la figure ET l'objet pca (qui contient toutes les infos)
    return fig, pca


def scree_plot_with_ideal_components(X_train_prep, variance_threshold=0.95, figsize=(10, 6), random_state=42):
    """
    Calcule la PCA, génère l'éboulis et détermine automatiquement 
    le nombre idéal de composantes selon le seuil de variance et le critère de Kaiser.
    """
    pca = PCA(random_state=random_state)
    pca.fit(X_train_prep)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    eigenvalues = pca.explained_variance_  # Valeurs propres

    # 1. Calcul selon le seuil de variance (ex: 95%)
    n_comp_var = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    # 2. Critère de Kaiser (valeur propre > 1)
    n_comp_kaiser = int(np.sum(eigenvalues > 1.0))

    # Génération du graphique
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(1, len(explained_variance) + 1),
           explained_variance, 
           alpha=0.5, align='center',
           label='Variance expliquée par composante', 
           color='#439cc8')
    
    ax.plot(range(1, len(cumulative_variance) + 1),
            cumulative_variance, 
            marker='o', linestyle='--',
            color='darkorange', 
            label='Variance expliquée cumulative')
            
    # Lignes de repère visuelles
    ax.axhline(y=variance_threshold, color='r', linestyle=':', label=f'Seuil {int(variance_threshold*100)}% ({n_comp_var} comp.)')
    ax.axvline(x=n_comp_kaiser, color='g', linestyle='-.', label=f'Critère de Kaiser ({n_comp_kaiser} comp.)')

    ax.set_xlabel('Composantes principales')
    ax.set_ylabel('Variance expliquée')
    ax.set_title('Éboulis des valeurs propres et Nombre Idéal de Composantes\n')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    return fig, n_comp_var, n_comp_kaiser