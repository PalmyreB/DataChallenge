Commandes UNIX utiles
=====================

## Se connecter au serveur

```
ssh -i ~/.ssh/KeyPair-c82a.pem  cloud@90.84.240.20
```

## Transférer un dossier local -> serveur

```bash
scp -i <dossier_du_fichier_.pem>/KeyPair-c82a.pem <fichier_à_copier> cloud@90.84.240.20:<dossier_de_destination>
```

