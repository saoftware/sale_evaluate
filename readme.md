# 1. Environnement virtuel

**Windows** : 
- Creer un environnement : python -m venv venv
- Activer l'environnement : venv\Scripts\activate

**Linux**
- Creer un environnement : python -m venv venv
- Activer l'environnement : venv\Scripts\activate


# 2. Installer les librairies

pip install -r requirements.txt

# 3. Lancer le serveur API
uvicorn main:app --reload