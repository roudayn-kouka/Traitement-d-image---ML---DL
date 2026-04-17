# Plant Disease Playground

Ce projet a ete migre vers une architecture React + Node.js + Express + MongoDB.

## Structure

```text
client/   Interface React (Vite)
server/   API Express + MongoDB + upload images
```

L'ancien code Python est encore present dans le depot comme reference technique, mais l'application principale passe maintenant par `client/` et `server/`.

## Fonctionnalites disponibles

- interface React pour uploader une image de feuille ;
- backend Express avec routes REST ;
- stockage MongoDB via Mongoose ;
- sauvegarde locale des images uploades ;
- historique des analyses avec suppression ;
- interface plus propre, responsive, et separee du backend ;
- inference reelle via le pipeline Python de traitement d'image existant.

## Lancer le backend

Copiez `server/.env.example` vers `server/.env`, puis configurez au minimum `MONGODB_URI`.

Par defaut, le backend utilise `../venv/Scripts/python.exe` pour appeler le pipeline Python existant. Si vous utilisez un autre environnement, adaptez `PYTHON_EXECUTABLE`.

```bash
cd server
npm install
npm run dev
```

## Lancer le frontend

```bash
cd client
npm install
npm run dev
```

Le frontend attend l'API sur `http://localhost:5000` et Vite sert l'interface sur `http://localhost:5173`.

## Lancer avec Docker Compose

```bash
docker compose up --build -d
```

Sous Windows, si le depot est dans OneDrive, Docker BuildKit peut echouer sur les fichiers `ReparsePoint`. Dans ce cas, utilisez le script PowerShell suivant depuis la racine du projet :

```powershell
.\docker-build.ps1
```

Le script cree une copie de travail normale dans `.docker-context/`, puis lance `docker compose up --build -d` depuis cette copie pour eviter l'erreur `invalid file request`.

Services exposes :

- frontend: `http://localhost:5173`
- backend: `http://localhost:5000`
- mongodb: `mongodb://localhost:27017`

Arret :

```bash
docker compose down
```

## API

- `GET /api/health`
- `GET /api/analyses`
- `POST /api/analyses`
- `DELETE /api/analyses/:id`

Le `POST /api/analyses` attend un formulaire `multipart/form-data` avec :

- `image`: fichier image
- `category`: texte optionnel

## Notes importantes

- Si `data/dataset` contient des sous-dossiers de classes avec des images, le backend utilise un modele `RandomForest` sur les features extraites.
- Si le dataset est vide ou insuffisant, le backend bascule automatiquement sur une inference heuristique basee sur la segmentation et les statistiques couleur de l'image.
