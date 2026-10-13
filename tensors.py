import torch

# Dans PyTorch, le tenseur est la brique de base universelle (building block). Tout est uniformisé sous forme de tenseur, qu'il s'agisse :
#  - d'un simple nombre (scalaire)
#  - d'une liste de nombres (vecteur)
#  - d'une image (matrice 2D)
#  - d'une vidéo (tenseur 4D)

# tenseur de dimension 0 (Le Scalaire)
t_scalar = torch.tensor(7.5)
print("Valeur :", t_scalar)
print("Forme (shape) :", t_scalar.shape)
# Valeur : tensor(7.5000)
# Forme (shape) : torch.Size([])


# tenseur de dimension 1 (Le Vecteur)
t_vector = torch.tensor([120.0, 4.0, 10.0])
print("Valeur :\n", t_vector)
print("Forme (shape) :", t_vector.shape)
# Forme (shape) : torch.Size([])
# Valeur :
#  tensor([120.,   4.,  10.])
# Forme (shape) : torch.Size([3])


# tenseur de dimension 2 (La Matrice)
# Exemple : un batch de 3 maisons, avec chacune 2 caractéristiques [surface, nombre de pièces]
t_matrix = torch.tensor([
    [80.0, 3.0], 
    [120.0, 4.0],
    [200.0, 6.0]
])
print("Valeur :\n", t_matrix)
print("Forme (shape) :", t_matrix.shape)  # torch.Size([3, 2]) -> 3 lignes, 2 colonnes
# Valeur :
# tensor([[ 80.,   3.],
#        [120.,   4.],
#        [200.,   6.]])
# Forme (shape) : torch.Size([3, 2])


# Forme : [batch_size (nombre de textes), sequence_length (nombre de mots), embedding_dim (caractéristiques par mot)]
t_3d = torch.tensor([
    # Texte 1 (3 mots, 4 caractéristiques chacun)
    [[1.0, 2.0, 3.0, 4.0], 
     [5.0, 6.0, 7.0, 8.0], 
     [9.0, 1.0, 2.0, 3.0]],
    
    # Texte 2 (3 mots, 4 caractéristiques chacun)
    [[4.0, 3.0, 2.0, 1.0], 
     [8.0, 7.0, 6.0, 5.0], 
     [3.0, 2.0, 1.0, 9.0]]
])
print("Valeur :\n", t_3d)
print("Forme (shape) :", t_3d.shape) 
# Valeur :
#  tensor([[[1., 2., 3., 4.],
#          [5., 6., 7., 8.],
#          [9., 1., 2., 3.]],
# 
#        [[4., 3., 2., 1.],
#          [8., 7., 6., 5.],
#          [3., 2., 1., 9.]]])
# Forme (shape) : torch.Size([2, 3, 4])
















