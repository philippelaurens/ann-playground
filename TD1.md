# Club 404 / TD1 : Atelier n8n & automatisation de l’information

> **Objectif** : mettre en place une chaîne d’automatisation de veille cybersécurité avec n8n (self‑hosted), NocoDB et un modèle IA gratuit (Groq / Mistral).  
> **Durée** : 1h
> **Public** : débutants à avancés, profils techniques et non techniques.

---

## Contexte & objectifs pédagogiques

- Comprendre ce qu’est un workflow d’automatisation (trigger → actions → conditions).
- Savoir installer et utiliser n8n en local (Docker).
- Utiliser une base no‑code (NocoDB) pour structurer une veille cyber.
- Exploiter un LLM gratuit (Groq / Mistral) pour résumer et prioriser des articles.
- Concevoir un mini‑bot de veille (Telegram ou webhook).

**Livrable attendu** : un workflow n8n fonctionnel qui :
- récupère des articles de veille ;
- les résume et les priorise avec l’IA ;
- les stocke dans NocoDB ;
- (optionnel) les renvoie via un bot Telegram.

---

## Prérequis techniques

- Machine avec Docker et Docker Compose installés.
- Un éditeur de texte (VS Code, Obsidian, etc.).
- Un compte :
  - [Groq Console](https://console.groq.com) **ou** [Mistral AI](https://console.mistral.ai) pour une API key.
  - (Optionnel) Un bot Telegram via @BotFather.

---

## 1) Installation de n8n (Docker Compose)

### 1.1 Créer le dossier de projet

```bash
mkdir n8n-compose && cd n8n-compose
```

### 1.2 Fichier `.env`

```env
N8N_HOST=n8n.local
N8N_PORT=5678
N8N_PROTOCOL=http
GENERIC_TIMEZONE=Europe/Paris
TZ=Europe/Paris
```

### 1.3 Fichier `compose.yaml`

```yaml
services:
  n8n:
    image: docker.n8n.io/n8nio/n8n:latest
    container_name: n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=${N8N_HOST}
      - N8N_PORT=${N8N_PORT}
      - N8N_PROTOCOL=${N8N_PROTOCOL}
      - GENERIC_TIMEZONE=${GENERIC_TIMEZONE}
      - TZ=${TZ}
    volumes:
      - n8n_data:/home/node/.n8n

volumes:
  n8n_data:
```

### 1.4 Lancer n8n

```bash
docker compose up -d
```

Accès : `http://localhost:5678` (ou `http://n8n.local:5678`).

> **Note** : pour un déploiement exposé, ajouter un reverse proxy (Caddy/Nginx) + TLS. Pour le TD, la version locale suffit.

---

## 2) Installation de NocoDB (base de veille)

### 2.1 Créer le dossier

```bash
mkdir nocodb-compose && cd nocodb-compose
```

### 2.2 Fichier `compose.yaml`

```yaml
services:
  nocodb:
    image: nocodb/nocodb:latest
    container_name: nocodb
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - nocodb_data:/usr/app/data

volumes:
  nocodb_data:
```

### 2.3 Lancer NocoDB

```bash
docker compose up -d
```

Accès : `http://localhost:8080`.

Créer un compte administrateur, puis une base `club404_veille`.

### 2.3 Table `articles`

Champs recommandés :

- `id` (auto)
- `titre` (text)
- `source` (text : RSS, blog, Twitter, etc.)
- `url` (text)
- `date_publi` (datetime)
- `tags` (text ou multi‑select : malware, phishing, vuln, CTI, etc.)
- `resume_ai` (text long)
- `priorite` (single select : faible / moyenne / haute)
- `statut` (single select : nouveau / à lire / traité / ignoré)

---

## 3) Configuration d’un provider IA gratuit

### 3.1 Groq (recommandé pour le PoC)

1. Créer un compte sur <https://console.groq.com>.
2. Générer une **API Key**.
3. Dans n8n :
   - **Credentials** → **OpenAI**.
   - API Key : ta clé Groq.
   - Base URL : `https://api.groq.com/openai/v1`
   - Modèle : ex. `llama-3.3-70b-versatile`.

### 3.2 Mistral AI (alternative)

1. Créer un compte sur <https://console.mistral.ai>.
2. Générer une **API Key**.
3. Dans n8n :
   - Utiliser le node **Mistral AI** ou **OpenAI Chat Model**.
   - Configurer avec la clé et le modèle choisi (ex. `open-mistral-7b`).

> Les quotas gratuits sont suffisants pour un TD avec quelques dizaines d’appels.

---

## 4) Scénario 1 / Veille cyber automatisée (RSS → NocoDB + IA)

### 4.1 Objectif

Construire un pipeline qui :

- lit des flux RSS d’actualités cyber ;
- filtre par mots‑clés ;
- résume et priorise avec l’IA ;
- enregistre dans NocoDB.

### 4.2 Étapes dans n8n

1. **Schedule Trigger**
   - Déclencheur : toutes les 6h (ou manuel pour le TD).
2. **RSS Read**
   - Ajouter des flux : blogs ANSSI, CERT, vendors, Reddit r/netsec, etc.
3. **Filter / If**
   - Conserver uniquement les articles contenant :
     - `ransomware`, `vulnerability`, `CVE`, `phishing`, `exploit`, etc.
4. **AI Chat Model (Groq / Mistral)**
   - Input : `titre` + `extrait`.
   - Prompt type :
     ```text
     Tu es un analyste cybersécurité.
     - Résume cet article en 2–3 phrases en français.
     - Propose 3 tags pertinents.
     - Estime la priorité (faible / moyenne / haute) selon l’impact potentiel pour une PME.

     Titre : {{ $json.titre }}
     Extrait : {{ $json.extrait }}
     ```
   - Récupérer : `resume_ai`, `tags`, `priorite`.
5. **NocoDB**
   - Action : **Create record** dans la table `articles`.
   - Mapper :
     - `titre`, `source`, `url`, `date_publi`
     - `tags`, `resume_ai`, `priorite`
     - `statut` = `nouveau`

### 4.3 Résultat attendu

Une base NocoDB remplie automatiquement avec des articles :
- résumés ;
- tagués ;
- priorisés.

---

## 5) Scénario 2 / Bot Telegram de veille

### 5.1 Objectif

Créer un bot qui :

- reçoit un lien d’article ;
- le résume et le priorise avec l’IA ;
- l’enregistre dans NocoDB ;
- renvoie un message structuré.

### 5.2 Création du bot Telegram

1. Ouvrir Telegram → @BotFather.
2. `/newbot` → choisir un nom et un username.
3. Récupérer le **token** du bot.

### 5.3 Workflow n8n

1. **Telegram Trigger** (ou **Telegram Bot** en polling)
   - Configurer avec le token.
2. **Filter**
   - Si le message contient une URL.
3. **HTTP Request**
   - Récupérer le titre + un extrait de la page (ou utiliser un service de scraping simple).
4. **AI Chat Model**
   - Même logique que scénario 1 : résumé, tags, priorité.
5. **NocoDB**
   - Créer un enregistrement dans `articles`.
6. **Telegram**
   - Répondre à l’utilisateur avec :
     - Titre
     - Résumé (2–3 lignes)
     - Priorité
     - Tags

> Variante : utiliser un **Webhook** (`POST /new-article`) + un petit script ou formulaire qui envoie `{ "url": "...", "source": "manuel" }`.

---

## 6) Connaissances abordées

### Automatisation & SOAR léger

- Workflow, trigger, action, condition.
- Différence entre automatisation simple et SOAR.
- Cas d’usage SOC : enrichissement d’IOC, création de tickets, notifications.

### Collecte & traitement de l’information

- Flux RSS, APIs, webhooks.
- Filtrage par mots‑clés.
- Normalisation des données (champs, tags, priorités).

### IA générative appliquée à la cybersécurité

- Utilisation d’un LLM pour :
  - résumer des articles ;
  - extraire des tags ;
  - estimer une priorité.
- Limites : hallucinations, biais, validation humaine.

### Bases de données no‑code / low‑code

- NocoDB comme « Airtable self‑hosted ».
- Schéma de table, types de champs, statuts de traitement.

### Notions d’architecture

- Où tourne n8n ? (Docker, VPS, homelab)
- Où tourne NocoDB ?
- Comment sécuriser l’accès (reverse proxy, auth, réseau isolé).

---

## 7) Déroulé type de la séance (1h)

- **5 min** : accueil, objectif, slogan Club 404.
- **5 min** : intro n8n + cas d’usage cybersécurité.
- **15 min** : démo live install n8n + NocoDB + Groq.
- **25 min** : atelier par "binômes" :
  - scénario veille RSS **ou** scénario bot Telegram.
- **5 min** : débrief, partage d’écrans, limites & pistes.
- **5 min** : prise de notes & idées de prolongement.

---

## 8) Pistes de prolongement

- Ajouter un node **HTTP Request** vers :
  - VirusTotal, AbuseIPDB, Shodan, etc. pour enrichir les IOC.
- Générer un rapport hebdomadaire automatique (PDF / Google Doc) des articles « haute priorité ».
- Connecter n8n à Wazuh / Suricata pour automatiser des actions sur alertes (notification, ticket, etc.).

---

## 9) Notes personnelles (à compléter pendant / après le TD)

- Points compris / flous :
- Commandes importantes à retenir :
- Idées d’amélioration pour mon propre lab / projet :
- Liens utiles découverts :
