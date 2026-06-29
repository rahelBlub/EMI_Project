# EMI_Projekt

## Inhaltsverzeichnis
- [Setup](#setup)
- [Ergebnistabelle](#ergebnistabelle)
- [Diskussion](#diskussion)


## Setup

Klonen des Repositorys:
````shell
git clone https://github.com/rahelBlub/EMI_Project.git
````

Virtuellen Enviroment erstellen und aktivieren:
````shell
# create virtual enviroment /.venv :
python3 -m venv .venv

#activate virtual enviroment:
source .venv/bin/activate
````

Benötigte Pakete installieren:
````shell
pip install -r .\requirements.txt
pip install git+https://github.com/openai/CLIP.git
````
CLIP package von GitHub: https://github.com/openai/CLIP/tree/main

Nach dem Klonen muss zuerst die `extract_features.py` gestartet werden.
Beim Starten wird geprüft ob das Datenset bereits im Verzeichnis ist, wenn nicht, wird dieses heruntergeladen.
Ansonsten wird das Datenset aus dem Verzeichnis `./data/dataset<Dataset_name_>` geladen. Es muss also kein Datenset manuell geladen werden

````bash
python extract_features.py
````
Anschließend muss die `models.py` gestartet werden. Das reicht einmalig, dann sind diese fest.

````bash
python models.py
````
Zum Schluss folgt dann noch die `train.py` mit dem Training und der Auswerung der Daten.

````bash
python train.py
````

## Ergebnistabelle


<img src="data/img/confusion_matrix_CrossAttentionFusion.png" width="500">

<table>
<tr style="font-size: larger">

  <th>
  Methode
  </th>
  
  <th>
  Macro-F1
  </th>
  
  <th>
  Accuracy
  </th>

</tr>

<tr>

  <th style="font-size: larger">
  Baseline Text only
  </th>
  
  <th>
  0.2362
  </th>
  
  <th>
  0.4050
  </th>

</tr>

<tr>

  <th style="font-size: larger">
  Baseline Image only
  </th>
  
  <th>
  0.1974
  </th>
  
  <th>
  0.3400
  </th>

</tr>

<tr>

  <th style="font-size: larger">
  EarlyFusion
  </th>
  
  <th>
  0.2078 ± 0.0064
  </th>
  
  <th>
  0.3583 ± 0.0118
  </th>

</tr>

<tr>

  <th style="font-size: larger">
  CrossAttentionFusion
  </th>
  
  <th>
  0.2302 ± 0.0107
  </th>
  
  <th>
  0.3667 ± 0.0062
  </th>

</tr>
</table>

**Modalitäts-Ablation (nur für bestes Modell - CrossAttention):** \
Ohne Text  : F1 = 0.2055  (Rückgang: +0.0247) \
Ohne Image : F1 = 0.2392  (Rückgang: -0.0091)

**Unimodale Baselines:** \
Unimodale Baseline (Text): F1=0.2362, Acc=0.4050 \
Unimodale Baseline (Bild): F1=0.1974, Acc=0.3400 

## Diskussion
tbd


