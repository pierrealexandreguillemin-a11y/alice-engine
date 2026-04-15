# Deploiement ALICE sur Oracle Cloud Infrastructure

> **Date**: 11 avril 2026
> **Statut**: EN ATTENTE — capacite ARM saturee a Marseille

---

## 1. Architecture Cible

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Frontend       │────▶│  Backend Node.js        │────▶│  ALICE Engine           │
│  Vercel         │     │  (chess-app)            │     │  OCI VM A1.Flex         │
│                 │     │                         │     │  FastAPI + ML models    │
└─────────────────┘     └─────────────────────────┘     └──────────┬──────────────┘
                                                                   │
                                                        ┌──────────┴──────────────┐
                                                        │  HuggingFace Hub        │
                                                        │  Pierrax/alice-engine   │
                                                        │  (model storage)        │
                                                        └─────────────────────────┘
```

Remplace l'architecture Render (voir `DEPLOIEMENT_RENDER.md`) par une VM OCI
permanente — plus de cold start, plus de limite 512 MB RAM.

---

## 2. Compte OCI

| Info | Valeur |
|------|--------|
| Tenancy | `pierrealexandreguillemin` |
| Email | `pierre.alexandre.guillemin@gmail.com` |
| Region home | `eu-marseille-1` (MRS) |
| Tier | Free (Always Free) |
| Cree le | 31 decembre 2025 |
| Console | https://cloud.oracle.com/?tenant=pierrealexandreguillemin |

### Authentification API

Config locale : `~/.oci/config`

```
[DEFAULT]
user=ocid1.user.oc1..aaaaaaaaxb5654nf6lyqemlx5pwcxdlqcvrqnzn3wvaypdybf4isn7bw747a
tenancy=ocid1.tenancy.oc1..aaaaaaaaugwnemzoo4dgaumposbz7ymvpagbrbl6cmnvljcjiaobnuw26x6q
region=eu-marseille-1
key_file=~/.oci/oci_api_key.pem
```

SSH key : `~/.oci/console_key.pub` (RSA, meme cle pour toutes les VMs).

---

## 3. Infra Existante

### Reseau

| Ressource | Nom | CIDR | OCID |
|-----------|-----|------|------|
| VCN | vcn-ffe | 10.0.0.0/16 | `ocid1.vcn.oc1.eu-marseille-1.amaaaaaarbqt6iiadonmm2vyxpvzp5ogactmhkge5gqxbzbewq7xj6sm4cea` |
| Subnet | public-subnet | 10.0.1.0/24 | `ocid1.subnet.oc1.eu-marseille-1.aaaaaaaa56wx3wdksiehh5exiksdjwfw7cckalpwixkaor5u5alv35ux6xbq` |
| Internet GW | igw-ffe | — | enabled |

### VMs existantes (E2.Micro free tier)

| VM | IP publique | IP privee | Role |
|----|-------------|-----------|------|
| ffe-scraper | 144.24.201.177 | 10.0.1.102 | Scraper FFE |
| ffe-scraper-2 | 84.235.227.48 | 10.0.1.142 | Scraper FFE |

OS : Oracle Linux 9, 1 OCPU AMD, 1 GB RAM chacune.

---

## 4. VM Alice Engine (a creer)

### Specs cibles

| Parametre | Valeur |
|-----------|--------|
| Shape | VM.Standard.A1.Flex |
| OCPU | 4 (Ampere Altra 3 GHz) |
| RAM | 24 GB |
| Disque | 50 GB boot |
| OS | Oracle Linux 9.7 aarch64 |
| Image | `ocid1.image.oc1.eu-marseille-1.aaaaaaaa5jmwo25wl7lrnrwrs3hgspxokuxhiys6jky5qfxwya7osvo3fg3a` |
| Hostname | alice-engine |
| Cout | Gratuit (Always Free A1 quota: 4 OCPU / 24 GB) |

### Quotas Free Tier (verifies 2026-04-11)

| Ressource | Limite | Utilise | Disponible |
|-----------|--------|---------|------------|
| A1 cores | 4 | 0 | 4 |
| A1 memory (GB) | 24 | 0 | 24 |
| E2.Micro | 2 | 2 | 0 |

### Probleme : "Out of host capacity"

La region Marseille est saturee en capacite ARM physique (probleme connu OCI
free tier, particulierement frequent sur les A1.Flex).

**Solutions** :
1. **Retry automatique** — script `C:\Dev\oci-create-alice.py` (retry toutes les 5 min)
2. **Autre region** — subscribe a eu-amsterdam-1 ou eu-frankfurt-1 depuis la console
3. **Upgrade PAYG** — passer en Pay-As-You-Go (toujours gratuit pour Always Free,
   mais meilleure priorite d'allocation)

---

## 5. Script de creation automatique

Fichier : `C:\Dev\oci-create-alice.py`

```bash
# Lancer le retry loop
python C:/Dev/oci-create-alice.py

# Le script retente toutes les 5 minutes
# Arret : Ctrl+C
# Succes : affiche l'IP publique et la commande SSH
```

Le script utilise le SDK Python OCI (`pip install oci`) et la config `~/.oci/config`.

---

## 6. Post-creation (TODO)

Une fois la VM creee :

1. **SSH** : `ssh opc@<IP_PUBLIQUE>` (user par defaut Oracle Linux = `opc`)
2. **Firewall** : ouvrir port 8000 (FastAPI) dans la Security List OCI
3. **Setup** :
   ```bash
   sudo dnf install -y python3.11 python3.11-pip git
   pip3.11 install fastapi uvicorn catboost xgboost lightgbm scikit-learn pandas
   git clone https://github.com/pierrealexandreguillemin-a11y/alice-engine.git
   ```
4. **Modeles** : telecharger depuis HuggingFace Hub `Pierrax/alice-engine`
5. **Systemd** : creer un service pour FastAPI (demarrage auto)
6. **HTTPS** : Caddy ou nginx + Let's Encrypt (ou WAA si load balancer)

---

## 7. Besoins memoire estimes

| Composant | RAM estimee |
|-----------|-------------|
| OS + systeme | ~500 MB |
| Python + FastAPI + deps | ~300 MB |
| CatBoost model (.cbm) | ~200 MB |
| XGBoost model (.ubj) | ~150 MB |
| LightGBM model (.txt) | ~100 MB |
| Feature store (parquets) | ~500 MB |
| Calibrators + encoders | ~50 MB |
| Marge | ~22 GB |
| **Total** | **~1.8 GB actif / 24 GB dispo** |

La VM 24 GB est largement surdimensionnee pour le serving. Le surplus
permet d'executer le feature refresh pipeline et des jobs ponctuels
sans risque d'OOM.
