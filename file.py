import torch

# Dans PyTorch, le tenseur est la brique de base universelle (building block). Tout est uniformisé sous forme de tenseur, qu'il s'agisse :
#  - d'un simple nombre (scalaire)
#  - d'une liste de nombres (vecteur)
#  - d'une image (matrice 2D)
#  - d'une vidéo (tenseur 4D)
# simples nombres emballés dans un tenseur, 
# ce qui leur permet de profiter de la machinerie de PyTorch (graphe de calcul, compatibilité GPU, etc.) tout en restant un scalaire.
x, y = torch.tensor(2.), torch.tensor(7.)

# requires_grad (booléen) indique à PyTorch s'il doit suivre les opérations faites sur ce tenseur 
#  pour calculer automatiquement les gradients plus tard.# 
w = torch.tensor(1., requires_grad=True)
b = torch.tensor(0., requires_grad=True)

# Loss function : Elle quantifie l'erreur de votre modèle.
#  Elle mesure à quel point la prédiction du modèle s'éloigne de la vraie valeur cible (y).
#  Squared Error
L = (w * x + b - y) ** 2

# L.backward() lance la rétropropagation du gradient (backpropagation).       
# -- Elle calcule les dérivées (gradients) : En partant de la perte L, elle parcourt le graphe de calcul à l'envers 
#   (grâce à la règle de dérivation en chaîne / chain rule) pour mesurer l'impact de chaque paramètre (w et b) sur l'erreur commise.
# -- Elle stocke les résultats : Elle enregistre ces gradients directement dans l'attribut .grad des tenseurs concernés 
#   (c'est-à-dire ceux qui ont l'attribut requires_grad=True).

# 
L.backward()
print(w.grad, b.grad)

L.backward()
print(w.grad, b.grad)