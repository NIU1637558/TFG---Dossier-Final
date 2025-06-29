# 📘 Optimització del Sistema de Gestió de Canvis en IT

## 🧠 Context i Motivació

En entorns IT corporatius, les sol·licituds de canvi (canvis) són operacions habituals i crítiques que afecten directament l’estabilitat i seguretat dels sistemes. Aquest projecte neix de la necessitat d'una empresa del sector tecnològic (ITnow) per **automatitzar la segona fase del seu sistema de gestió de canvis**, que fins ara depenia totalment de la supervisió humana. L’objectiu és reduir la càrrega operativa mitjançant tècniques d’aprenentatge automàtic (Machine Learning).

---

## 🎯 Objectius

- Avaluar la **viabilitat tècnica i pràctica** de l'automatització.
- Detectar i solucionar problemes de qualitat de dades.
- Construir un **pipeline complet de ML** capaç d’emular la decisió humana.
- Experimentar amb **embeddings i arquitectures avançades**.
- Proposar una solució **desplegable i escalable** per a entorns productius.

---

## 🏗️ Funcionament del Sistema Actual

El sistema corporatiu de validació de canvis té dues fases:
1. **Filtre automàtic (RPA):** decisions simples sobre camps categòrics.
2. **Revisió humana:** anàlisi detallada del canvi, inclòs el text lliure.

> 🎯 L’objectiu és automatitzar la segona fase, mantenint la intervenció humana només per casos amb alta incertesa.

---

## 📊 Dades Utilitzades

- **110.000 registres originals**, filtrats a 80.000 (tipus "Normal").
- **Variables**:
  - 14 camps categòrics
  - 3 camps de text lliure: `descripció`, `motiu`, `impacte`
- **Tractament**:
  - Traducció automàtica (portuguès → català)
  - Anonimització
  - Generació d’etiquetes a partir d'estats de MAXIMO

---

## 🧪 Metodologia

1. Anàlisi i neteja de dades
2. Generació d’etiquetes a partir de lògica empresarial
3. Models inicials: Random Forest, XGBoost, MLP
4. Millores progressives:
   - Dropout, BatchNorm, L2, Weighted Loss
   - Bagging (20 estimadors)
   - Arquitectura **MLP + Attention**
5. Embeddings provats:
   - Word2Vec, Doc2Vec, MiniLM, DistilBERT
   - Reducció amb autoencoders
   - Finetuning amb corpus específic
6. Model combinat **(Ensemble amb Sub-CNN)**

---

## 📈 Resultats

- **Millora acumulada**: +10.5% respecte al model base
- Millor model:
  - `Recall TRUE`: 0.812
  - `Recall FALSE`: 0.810
  - `Precision`: 0.805
- **Generalització robusta** en conjunts RPA i acceptable en Human

---

## ⚠️ Limitacions

- **Etiquetes corruptes**: sistema actual sobreescriu canvis rebutjats.
- **Solució proposada**: monitorització en temps real amb script per capturar estats reals abans de modificació.
- **Estimació**: recollir 10.000 mostres en ~12 mesos.

---

## 📂 Estructura del Repositori

📁 code/
└── /classificators/: Tots els codis emprats per a entrenar i provar els classificadors
└── /data_preparation/: Conté l'arxiu usat per a realitzar la traducció. NOTA: La majoria de arxius usats pel preprocessat sópn provats deguts a estar desenvolupats amb l'empresa.
└── /data_quality/: Conté el codi i els resultats del 1r estudi de rellevància de text lliure.
└── /embedder/: Conté els codis, reusltats, i models precarregats dels Autoencoders usats pels Embedders.
NOTA IMPORTANT: Dins de /classificatoirs/DL/code/ s'inclou la explicació de cadascun dels arxius.

📁 models/
└── /hybrid_models/: Conté els models entrenats amb l'arquitrectura del model combinat d'ensamblatge ponderat. Les versioins finals son els acabats en 42.
└── /hybrid_neurons_pretrained/: Inclou els models preentrenats usats en el model combinat d'ensamblatgte ponderat que s'entrena amb models preentrenats.
└── /hybrid_data_experiment/: Inclou els models combinats d'ensamblatgte ponderat entrenats en diferents particions de dades pel experiment de rellevància 2.
└── /classifiers/: Conté TOTS els models generats en entrenaments individuals.

📁 data/
└── Conté els diversos conjunts de dades usats, el usat finalment és el CH_Total2.csv

📁 results/
└── /DataQuality2/: Conté els resultats del experiment de rellevància 2 amb diferent variable objectiu,
└── /embedder_experiment/ Conté els resultats del experiment de rellevància 2 amb la variable objectiu REJECTED i el millor model individual.
└── {metric_results}.csv: Diversos .csv amb els resultats dels experiments de creuament dels conjunts de dades