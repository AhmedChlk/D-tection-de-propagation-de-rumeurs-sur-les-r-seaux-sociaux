# Projet : Détection de Rumeurs (FakeNewsNet)

Ce document définit les standards et les directives obligatoires pour le développement de ce projet. Il prévaut sur toute autre instruction générale.

## 1. Stack Technologique
- **Langage** : Python
- **Frameworks Deep Learning** : PyTorch 
- **Traitement de données** : Pandas, Scikit-learn
- **NLP** : NLTK, Spacy

## 2. Structure du Projet (Processus KDD)
Le code doit être strictement séparé selon les étapes du processus Knowledge Discovery in Databases :
1. **Sélection** : Acquisition et filtrage des données brutes (FakeNewsNet).
2. **Prétraitement** : Nettoyage (bruit, valeurs manquantes, tokenisation).
3. **Transformation** : Feature engineering, vectorisation (Word2Vec, BERT, etc.), normalisation.
4. **Data Mining** : Architecture des modèles de Deep Learning et entraînement.
5. **Interprétation/Évaluation** : Analyse des résultats, métriques de performance, visualisation.

## 3. Règles de Deep Learning Obligatoires
Toutes les implémentations de réseaux de neurones doivent respecter les principes suivants :

- **Initialisation des Poids** : Utilisation systématique de l'initialisation de **He (Kaiming)** pour toutes les couches utilisant ReLU ou ses variantes.
- **Stabilisation** : Intégration de la **Normalisation par lots (Batch Normalization)** après les couches linéaires/convolutives pour stabiliser les activations.
- **Régularisation (Anti-Overfitting)** :
  - Utilisation de couches **Dropout** entre les couches denses.
  - Implémentation de l'**arrêt prématuré (Early Stopping)** basé sur la perte de validation.
- **Fonction de Perte** : Utilisation de l'**Entropie Croisée (Cross-Entropy)** pour les tâches de classification.
- **Optimiseur** : Utilisation de l'optimiseur **Adam** (Adaptive Moment Estimation) par défaut.
