---

## 📁 Estructura del Projecte

### 🗂️ Carpetes Principals

- `/functions`  
  Conté funcions auxiliars per:
  - preprocessat de dades  
  - entrenament de models  
  - test de rendiment  
  - tractament d’inputs/outputs

- `/models_arquitectres`  
  Conté les definicions de les diferents arquitectures provades:
  - MLP bàsic
  - MLP amb Attention
  - Encoder
  - Model combinat amb Sub-CNN

- `/model_hybrid_test`  
  Conté codi per provar el **model híbrid** (combinació de múltiples arquitectures i embeddings).

- `/resto_pruebas`  
  Conté la resta de codis utilitzats per fer proves experimentals al llarg del projecte:
  - optimitzacions
  - tuning de paràmetres
  - proves de balanceig
  - test de embeddings, etc.

---

### 🔧 Codis Principals

- `PIPELINE.py`  
  Script principal per **executar la pipeline final**, carregant el millor model i rebent l’ID anonimitzat d’un canvi.  
  ➤ Ús: aplicable a predicció directa en entorns reals.

- `DataQuality_classif2.py`  
  Prova la capacitat predictiva del model sobre **dades amb variables de qualitat diferent**.  
  ➤ Permet estudiar robustesa davant problemes reals.

- `MLP2_authorcross.py`  
  Experiment de **validació creuada** per orígens de dades (RPA vs Human).  
  ➤ Permet analitzar la transferència de coneixement entre conjunts.

- `MLP2_main.py`  
  Entrenament bàsic d’un model individual (indicat al codi).  
  ➤ És el punt d’inici per entrenar qualsevol arquitectura simple.

- `MLP2_main_bagging2.py`  
  Entrena un model amb **bagging ensemble** (20 estimadors).  
  ➤ Redueix l’overfitting i augmenta estabilitat entre execucions.

- `hybrid_main2.py`  
  Entrenament del **model híbrid**, que combina les prediccions de múltiples models individuals usant Sub-CNN.  
  ➤ Representa l’estratègia més robusta i final del projecte.

---
