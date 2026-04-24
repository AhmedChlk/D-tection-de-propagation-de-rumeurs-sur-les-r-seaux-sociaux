---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
  @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

  /* ANIMATIONS KEYFRAMES */
  @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

  /* FOND GÉNÉRAL DES SLIDES (Premium Light Tech) */
  section {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: radial-gradient(circle at 0% 0%, #f1f5f9 0%, #ffffff 100%);
    background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
    background-size: 40px 40px;
    color: #334155;
    font-size: 20px;
    padding: 60px 70px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  /* SLIDES SPÉCIALES (Titre & Fin) */
  section.dark-slide {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
    color: #f8fafc;
  }
  section.dark-slide h1 { color: white; border: none; }

  section.title-slide h1 {
    font-size: 2.5em; font-weight: 800; line-height: 1.2;
    background: linear-gradient(to right, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: fadeInUp 1s ease-out forwards; margin-bottom: 20px; text-align: center;
  }

  /* TITRES */
  h1 { color: #0f172a; font-size: 1.8em; font-weight: 800; margin-bottom: 30px; animation: fadeInUp 0.6s ease-out forwards; }
  h2 { color: #1e293b; border-bottom: 3px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 30px; font-size: 1.6em; font-weight: 700; display: flex; align-items: center; animation: fadeInUp 0.6s ease-out forwards; }
  h3 { color: #3b82f6; font-size: 1.1em; font-weight: 700; margin-top: 0; margin-bottom: 15px; display: flex; align-items: center; }
  h3 i { margin-right: 10px; font-size: 1.2em; }

  /* MISE EN PAGE ET GLASSMORPHISM CARDS */
  .columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 30px; align-items: stretch; }
  .columns-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; text-align: center;}

  .card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-top: 4px solid #3b82f6;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05);
    animation: fadeInUp 0.8s ease-out 0.2s forwards; opacity: 0;
  }

  .columns > div:nth-child(2) .card { animation-delay: 0.4s; }
  .card-danger { border-top-color: #f43f5e; }
  .card-success { border-top-color: #10b981; }

  /* LISTES ET TEXTE ANIMÉS */
  ul, ol { padding-left: 20px; margin: 0; }
  p { line-height: 1.5; margin-top: 0; }
  li { margin-bottom: 12px; line-height: 1.5; animation: fadeInUp 0.5s ease-out forwards; opacity: 0; }
  li:nth-child(1) { animation-delay: 0.3s; } li:nth-child(2) { animation-delay: 0.5s; } li:nth-child(3) { animation-delay: 0.7s; }

  /* ICÔNES ET COULEURS */
  i.icon-main { color: #3b82f6; width: 35px; text-align: center; margin-right: 10px; }
  .text-red { color: #f43f5e; } .text-green { color: #10b981; } .text-blue { color: #3b82f6; } .text-orange { color: #f59e0b; }

  /* SCHÉMAS D'ARCHITECTURE (CSS) */
  .arch-flow { display: flex; align-items: center; justify-content: space-between; margin-top: 20px; font-size: 0.8em; }
  .arch-box { background: #1e293b; color: white; padding: 15px 20px; border-radius: 8px; text-align: center; font-weight: 600; flex: 1; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  .arch-arrow { color: #94a3b8; font-size: 1.5em; margin: 0 10px; }
  .arch-box.frozen { background: #f1f5f9; color: #64748b; border: 2px dashed #cbd5e1; box-shadow: none; }
  .arch-box.active { background: #eff6ff; color: #1e40af; border: 2px solid #3b82f6; }

  /* BIG STATS */
  .big-stat { text-align: center; animation: fadeInUp 0.8s ease forwards; opacity: 0; }
  .big-stat-number { font-size: 3em; font-weight: 800; line-height: 1; margin-bottom: 5px; }
  .big-stat-label { font-size: 0.8em; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
---

<div style="text-transform: uppercase; letter-spacing: 3px; font-size: 0.8em; font-weight: 600; color: #94a3b8; text-align: center; margin-bottom: 30px; animation: fadeInUp 1s ease-out 0.3s forwards; opacity: 0;">
  Projet de Fin de Module — Deep Learning
</div>

<h1>Détection de propagation de rumeurs<br>sur les réseaux sociaux</h1>

<div style="display: flex; justify-content: center; gap: 60px; margin-top: 60px; text-align: left; animation: fadeInUp 1s ease-out 0.6s forwards; opacity: 0;">
  <div>
    <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 5px;"><i class="fa-solid fa-user-graduate"></i> Présenté par</p>
    <p style="font-weight: 700; font-size: 1.2em; margin: 0; color: white;">FRENDI Ilham</p>
    <p style="font-weight: 700; font-size: 1.2em; margin: 0; color: white;">Ahmed [Ton Nom]</p>
  </div>
  <div style="width: 2px; background: #475569; border-radius: 2px;"></div>
  <div>
    <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 5px;"><i class="fa-solid fa-chalkboard-user"></i> Sous la direction de</p>
    <p style="font-weight: 700; font-size: 1.2em; margin: 0; color: white;">Dr. Khedoudja BOUAFIA</p>
  </div>
</div>

<div style="margin-top: 80px; font-size: 0.8em; color: #64748b; font-weight: 500; text-align: center; animation: fadeInUp 1s ease-out 0.9s forwards; opacity: 0;">
  Université Abderrahmane-Mira de Béjaïa | Master 1 Intelligence Artificielle
</div>

---

## <i class="fa-solid fa-crosshairs icon-main"></i> Contexte et Problématique

<div class="columns">
  <div>
    <h3 style="color: #0f172a;"><i class="fa-solid fa-globe"></i> Le fléau de la désinformation</h3>
    <ul>
      <li><strong>Le paradigme Big Data :</strong> Vélocité et Volume extrêmes de l'information sur les réseaux sociaux.</li>
      <li><strong>Enjeux sociétaux :</strong> Manipulation de l'opinion, crises sanitaires et déstabilisation démocratique.</li>
      <li><strong>Verrous NLP :</strong> Textes limités (280 caractères), syntaxe dégradée et bruit structurel important.</li>
    </ul>
  </div>
  <div class="card">
    <h3><i class="fa-solid fa-bullseye"></i> Objectif Scientifique</h3>
    <p>Concevoir une architecture <strong>Deep Learning</strong> capable de discriminer la véracité d'une information en s'appuyant exclusivement sur le signal sémantique d'un tweet source.</p>
    <div style="margin-top: 20px; padding: 15px; background: #eff6ff; border-radius: 8px; font-size: 0.85em; color: #1e40af; border-left: 4px solid #3b82f6;">
      <i class="fa-solid fa-lightbulb"></i> <em>Transition d'une ingénierie de caractéristiques manuelle vers un apprentissage de représentations automatisé.</em>
    </div>
  </div>
</div>

---

## <i class="fa-solid fa-triangle-exclamation icon-main"></i> Phase 1 : L'échec instructif (FakeNewsNet)

Avant d'atteindre le modèle optimal, une expérimentation a été menée sur <em>FakeNewsNet</em> (articles journalistiques factuels de type <em>PolitiFact/BuzzFeed</em>).

<div class="columns-3" style="margin-top: 30px;">
  <div class="card card-danger big-stat" style="animation-delay: 0.3s;">
    <div class="big-stat-number text-red">37.5%</div>
    <div class="big-stat-label">Accuracy TextCNN</div>
  </div>
  <div class="card card-danger big-stat" style="animation-delay: 0.5s;">
    <div class="big-stat-number text-red">53.1%</div>
    <div class="big-stat-label">Accuracy DistilBERT</div>
  </div>
  <div style="text-align: left; padding: 20px; display: flex; flex-direction: column; justify-content: center; animation: fadeInUp 0.8s ease forwards 0.7s; opacity: 0;">
    <h3 style="color: #0f172a;"><i class="fa-solid fa-stethoscope"></i> Diagnostic</h3>
    <p style="font-size: 0.85em;"><strong>1. Overfitting sévère :</strong> Corpus restreint (~400 articles).</p>
    <p style="font-size: 0.85em;"><strong>2. Biais Sémantique :</strong> Les articles factuels sont dépourvus des marqueurs cognitifs de la rumeur.</p>
  </div>
</div>

---

## <i class="fa-solid fa-database icon-main"></i> Phase 2 : Le pivot stratégique (Twitter15/16)

<p style="margin-bottom: 25px;">Pour capturer la dynamique des réseaux sociaux, l'étude s'est réorientée vers un corpus académique nativement conçu pour l'analyse de propagation de rumeurs.</p>

<div class="columns">
  <div>
    <div class="card card-success" style="margin-bottom: 20px; padding: 20px;">
      <div style="display: flex; align-items: center; gap: 15px;">
        <i class="fa-brands fa-twitter" style="font-size: 2.5em; color: #1da1f2;"></i>
        <div>
          <div style="font-size: 1.5em; font-weight: 800; color: #0f172a;">2 308 Tweets</div>
          <div style="font-size: 0.85em; color: #64748b; font-weight: 600; text-transform: uppercase;">Racines de cascades d'information</div>
        </div>
      </div>
    </div>
    <p style="font-size: 0.9em;"><i class="fa-solid fa-check text-green"></i> <strong>Bénéfice :</strong> Échantillon représentatif du langage social, classes parfaitement équilibrées.</p>
  </div>
  <div class="card">
    <h3><i class="fa-solid fa-list-check"></i> Classification en 4 états</h3>
    <ul style="list-style-type: none; padding-left: 0; margin-top: 15px;">
      <li style="display: flex; align-items: center;"><i class="fa-solid fa-circle-check text-green"></i> <strong>True (579) :</strong> Rumeur avérée exacte</li>
      <li style="display: flex; align-items: center;"><i class="fa-solid fa-circle-xmark text-red"></i> <strong>False (575) :</strong> Rumeur avérée fausse</li>
      <li style="display: flex; align-items: center;"><i class="fa-solid fa-circle-question text-orange"></i> <strong>Unverified (575) :</strong> Non confirmée</li>
      <li style="display: flex; align-items: center;"><i class="fa-solid fa-info-circle text-blue"></i> <strong>Non-Rumor (579) :</strong> Information standard</li>
    </ul>
  </div>
</div>

---

## <i class="fa-solid fa-filter icon-main"></i> Prétraitement et Nettoyage (NLP)

<p style="margin-bottom: 25px;">Afin d'isoler le signal sémantique et de réduire l'entropie, un pipeline de normalisation strict a été exécuté sur les données brutes :</p>

<div class="columns">
  <div class="card">
    <h3><i class="fa-solid fa-gears"></i> Séquence de Normalisation</h3>
    <ol style="font-size: 0.95em; padding-left: 20px;">
      <li><strong>Anonymisation :</strong> Purge des entités (<code>@user</code>).</li>
      <li><strong>Nettoyage Web :</strong> Extraction des hyperliens.</li>
      <li><strong>Filtrage :</strong> Éradication de la ponctuation résiduelle.</li>
      <li><strong>Tokenisation :</strong> Génération du vocabulaire par fréquence.</li>
    </ol>
  </div>
  <div>
    <h3 style="color: #0f172a;"><i class="fa-solid fa-magnifying-glass"></i> Justification Analytique</h3>
    <p style="font-size: 0.95em;">Dans un espace restreint de 280 caractères, le maintien d'URLs ou de mentions provoque une sparsité matricielle critique.</p>
    <p style="font-size: 0.95em;">Ce nettoyage force le réseau à converger vers la sémantique pure de la rumeur.</p>
  </div>
</div>

---

## <i class="fa-solid fa-layer-group icon-main"></i> Architecture 1 : TextCNN

<p style="margin-bottom: 25px;">Transposition du paradigme des réseaux convolutifs vers l'analyse séquentielle (1D).</p>

<div class="arch-flow">
  <div class="arch-box" style="background: #e2e8f0; color: #0f172a;">Tweet<br>Tokenisé</div>
  <i class="fa-solid fa-arrow-right arch-arrow"></i>
  <div class="arch-box" style="background: #3b82f6;">Embedding<br>Layer</div>
  <i class="fa-solid fa-arrow-right arch-arrow"></i>
  <div style="display: flex; flex-direction: column; gap: 5px; flex: 1.5;">
    <div class="arch-box" style="background: #0f172a; padding: 5px;">Filtre Conv1D (k=3)</div>
    <div class="arch-box" style="background: #0f172a; padding: 5px;">Filtre Conv1D (k=4)</div>
    <div class="arch-box" style="background: #0f172a; padding: 5px;">Filtre Conv1D (k=5)</div>
  </div>
  <i class="fa-solid fa-arrow-right arch-arrow"></i>
  <div class="arch-box" style="background: #f59e0b;">1D Max<br>Pooling</div>
  <i class="fa-solid fa-arrow-right arch-arrow"></i>
  <div class="arch-box" style="background: #ef4444;">MLP<br>(4 Classes)</div>
</div>

<div class="card" style="margin-top: 30px; padding: 20px;">
  <p style="margin: 0; font-size: 0.9em;"><strong><i class="fa-solid fa-microscope text-blue"></i> Intuition Mathématique :</strong> Les filtres convolutifs agissent comme des détecteurs de <strong>N-grammes spatiaux</strong>. Ils identifient des patterns lexicaux invariants dans l'espace séquentiel.</p>
</div>

---

## <i class="fa-solid fa-diagram-project icon-main"></i> Architecture 2 : DistilBERT (Transfer Learning)

<p style="margin-bottom: 25px;">Exploitation d'un modèle basé sur l'Attention, pré-entraîné sur des corpus massifs.</p>

<div class="columns">
  <div class="card card-danger">
    <h3><i class="fa-solid fa-triangle-exclamation text-red"></i> Le Risque Structurel</h3>
    <p style="font-size: 0.95em;">DistilBERT embarque <strong>66 Millions de paramètres</strong>. Son optimisation directe sur un micro-corpus induit inévitablement un <em>catastrophic forgetting</em> ou un <em>mode collapse</em>.</p>
  </div>
  <div class="card card-success">
    <h3><i class="fa-solid fa-sliders text-green"></i> Stratégie : Fine-Tuning</h3>
    <div class="arch-flow" style="flex-direction: column; gap: 10px; margin-top: 10px;">
      <div class="arch-box active" style="width: 100%;"><i class="fa-solid fa-bolt"></i> Classifieur MLP (Entraîné)</div>
      <div class="arch-box active" style="width: 100%;"><i class="fa-solid fa-bolt"></i> 2 Couches Hautes (Ajustées)</div>
      <div class="arch-box frozen" style="width: 100%;"><i class="fa-solid fa-lock"></i> 4 Couches Basses (Gelées)</div>
    </div>
  </div>
</div>

---

## <i class="fa-solid fa-chart-area icon-main"></i> Analyse de la Convergence (Courbes d'Apprentissage)

<p style="margin-bottom: 10px;">Visualisation de la stabilisation du modèle TextCNN sur 20 époques d'entraînement.</p>

<div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;">
  <svg viewBox="0 0 800 280" style="width: 90%; height: auto; max-height: 300px; margin: 0 auto; display: block; font-family: 'Plus Jakarta Sans', sans-serif;">
    <line x1="60" y1="240" x2="750" y2="240" stroke="#cbd5e1" stroke-width="2"/> <line x1="60" y1="20" x2="60" y2="240" stroke="#cbd5e1" stroke-width="2"/> <line x1="60" y1="185" x2="750" y2="185" stroke="#f1f5f9" stroke-width="1"/> <line x1="60" y1="130" x2="750" y2="130" stroke="#f1f5f9" stroke-width="1"/> <line x1="60" y1="75" x2="750" y2="75" stroke="#f1f5f9" stroke-width="1"/> <path d="M 60 210 Q 120 150 180 90 T 300 77 T 500 76 T 750 75" fill="none" stroke="#3b82f6" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    
    <path d="M 60 176 Q 120 150 180 145 T 300 135 T 500 125 T 750 125" fill="none" stroke="#10b981" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    
    <text x="40" y="245" font-size="12" fill="#64748b" text-anchor="end">0%</text>
    <text x="40" y="189" font-size="12" fill="#64748b" text-anchor="end">50%</text>
    <text x="40" y="134" font-size="12" fill="#64748b" text-anchor="end">75%</text>
    <text x="40" y="79" font-size="12" fill="#64748b" text-anchor="end">100%</text>
    <text x="740" y="260" font-size="12" fill="#64748b">Époques (20)</text>
    
    <circle cx="600" cy="30" r="6" fill="#3b82f6"/> <text x="615" y="34" font-size="13" fill="#334155" font-weight="600">Train Accuracy</text>
    <circle cx="600" cy="55" r="6" fill="#10b981"/> <text x="615" y="59" font-size="13" fill="#334155" font-weight="600">Validation Accuracy</text>
  </svg>
</div>

<p style="font-size: 0.85em; text-align: center; color: #64748b; margin-top: 10px;">
  <i class="fa-solid fa-chart-line"></i> L'absence de divergence prononcée entre l'entraînement et la validation prouve l'absence d'overfitting.
</p>

---

## <i class="fa-solid fa-chart-simple icon-main"></i> Résultats Expérimentaux Globaux

<p>Évaluation de la capacité de généralisation sur un ensemble de test aveugle (15% du corpus).</p>

<div style="background: white; border-radius: 12px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px;">
  <svg viewBox="0 0 800 200" style="width: 100%; height: auto; font-family: 'Plus Jakarta Sans', sans-serif;">
    <text x="180" y="45" font-size="14" fill="#1e293b" font-weight="600" text-anchor="end">Baseline Aléatoire</text>
    <rect x="200" y="25" width="137.5" height="26" rx="4" fill="#94a3b8"/> <text x="345" y="43" font-size="14" fill="#64748b" font-weight="700">25.00%</text>

    <text x="180" y="105" font-size="14" fill="#1e293b" font-weight="600" text-anchor="end">DistilBERT (Fine-Tuned)</text>
    <rect x="200" y="85" width="434.28" height="26" rx="4" fill="#3b82f6">
      <animate attributeName="width" from="0" to="434.28" dur="1s" fill="freeze" />
    </rect> <text x="642" y="103" font-size="14" fill="#2563eb" font-weight="700">78.96%</text>

    <text x="180" y="165" font-size="14" fill="#1e293b" font-weight="600" text-anchor="end">TextCNN (Optimal)</text>
    <rect x="200" y="145" width="437.47" height="26" rx="4" fill="#10b981">
      <animate attributeName="width" from="0" to="437.47" dur="1s" fill="freeze" />
    </rect> <text x="645" y="163" font-size="14" fill="#059669" font-weight="800">79.54%</text>

  </svg>
</div>

---

## <i class="fa-solid fa-scale-balanced icon-main"></i> Analyse Dimensionnelle : CNN vs BERT

<div class="columns">
  <div>
    <h3 style="color: #0f172a;"><i class="fa-solid fa-bolt"></i> Observation Contre-Intuitive</h3>
    <p style="font-size: 0.95em;">Le réseau TextCNN, d'architecture structurellement superficielle, surperforme de manière marginale DistilBERT, pourtant établi comme le standard industriel actuel du NLP.</p>
  </div>
  <div class="card">
    <h3><i class="fa-solid fa-brain"></i> Interprétation Théorique</h3>
    <p style="font-size: 0.95em;">La contrainte spatiale justifie ce résultat : <strong>les données d'entrée sont d'une extrême brièveté</strong> (Tweets).</p>
    <p style="font-size: 0.95em; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
      Sur ces micro-séquences, l'extraction spatiale locale (CNN) s'avère algorithmiquement aussi résiliente que le calcul lourd des dépendances globales (Mécanisme d'Attention).
    </p>
  </div>
</div>

---

## <i class="fa-solid fa-flag-checkered icon-main"></i> Conclusion et Travaux Futurs

<div class="columns">
  <div>
    <h3 style="color: #0f172a;"><i class="fa-solid fa-list-check"></i> Bilan des contributions</h3>
    <ul style="font-size: 0.95em;">
      <li><strong>Faisabilité validée :</strong> Détection automatisée à ~80% de précision.</li>
      <li><strong>Data-Centric AI :</strong> Preuve que la qualité sémantique du corpus prime sur la profondeur brute du réseau.</li>
      <li><strong>Efficience :</strong> Supériorité computationnelle des Convolutions 1D pour l'analyse de micro-textes.</li>
    </ul>
  </div>
  <div class="card">
    <h3><i class="fa-solid fa-rocket"></i> Perspectives de Recherche</h3>
    <ul style="font-size: 0.95em; list-style: none; padding-left: 0;">
      <li style="margin-bottom: 20px;">
        <strong><i class="fa-solid fa-project-diagram text-blue"></i> Topologie de réseau (GNN)</strong><br>
        Modéliser mathématiquement la structure des arbres de propagation (qui retweete qui).
      </li>
      <li>
        <strong><i class="fa-solid fa-clock-rotate-left text-blue"></i> Dynamique Temporelle (RNN)</strong><br>
        Intégrer des LSTM pour capter la vélocité et la chronologie d'apparition des interactions.
      </li>
    </ul>
  </div>
</div>

---

<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; text-align: center;">

<i class="fa-solid fa-quote-left" style="font-size: 4em; color: rgba(255,255,255,0.1); margin-bottom: 30px; animation: fadeInUp 1s ease-out forwards;"></i>

  <h1 style="font-size: 4em; margin-bottom: 20px; animation: fadeInUp 1s ease-out 0.2s forwards; opacity: 0;">Merci de votre attention</h1>
  
  <p style="font-size: 1.5em; color: #94a3b8; font-weight: 500; animation: fadeInUp 1s ease-out 0.4s forwards; opacity: 0;">Nous sommes à votre disposition pour toute question.</p>

  <div style="margin-top: 80px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 1em; color: #cbd5e1; font-weight: 600; letter-spacing: 2px; animation: fadeInUp 1s ease-out 0.6s forwards; opacity: 0; width: 60%;">
    FRENDI Ilham & Ahmed [Ton Nom] | Master 1 IA
  </div>

</div>
