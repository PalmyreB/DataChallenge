Explications sur les matrices
===

# Format

Matrice de théoriquement 200000 lignes (nombres de softwares/malwares) et deux colonnes :
- liste des couples (parent, enfant)
- liste des lignes des *behavior_sequence.txt*

# Fichiers

## matrix_training_i.npy
Matrice créée avec ```np.save('matrix_training_i.npy')```
20000 lignes

Se charge avec :
```py
import numpy as np
M = np.load('matrix_training_i.npy')[1:,:]
```

## matrix.raw
Matrice créée avec ```matrix.dump('matrix.raw')``` (numpy)

Se charge avec :
```py
import numpy as np
M = np.load('matrix.raw')
```

## sample.raw
Sous-matrice des mille premières lignes

Se charge avec :
```py
import numpy as np
M = np.load('sample.raw')
```

## matrix.txt
Matrice au format texte

Se recopie éventuellement avec (à tester) :
```py
with open('matrix.txt', 'r') as file:
    M = eval(file.read().replace('\n', ''))
```
