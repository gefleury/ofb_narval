# Répertoire des données

Le présent répertoire `data_example` fournit la structure des données et quelques exemples de PDFs à analyser.  



## Les données en entrée 
Dans le répertoire `data_example/input/` :
- **Sous-répertoire `pdfs/` :**   
Contient des rapports RPQS au format PDF collectés sur le [site de SISPEA](https://www.services.eaufrance.fr/).

- **Sous-répertoire `sispea_indic_values/` :**  
 Contient pour chaque PDF les valeurs correspondantes des indicateurs *saisies sur SISPEA* par les collectivités, au format csv. Elles peuvent différer des valeurs inscrites dans le PDF (coquilles, arrondis, date de saisie ultérieure, etc...).  
Source : https://services.eaufrance.fr/pro/telechargement  

- **Sous-répertoire `sispea_vs_pdf_indic_values/` :**   
 Contient les fichiers de labélisation des rapports RPQS contenus dans `pdfs/`, utilisés comme référence pour l'évaluation. La version `v2` est à privilégier et est utilisée ci-après pour l'évaluation.   

- **Fichier `indicateurs_v*.csv` :**  
Liste des indicateurs SISPEA, avec notamment leur code, leur unité, leurs bornes min/max et des instructions spécifiques pour les prompts.  

- **Fichier `question_keyword_v*.csv` :**  
Liste de questions à poser au modèle de langage (eg Llama3.1) pour chaque indicateur. Chaque question est associée à un mot-clé permettant d'identifier dans un PDF les pages pertinentes pour répondre à cette question.

- **Fichier `rpqs_eval_list_1.csv` :**  
Liste de PDFs à analyser pour un run du notebook [run_full_pipeline.ipynb](../notebooks/run_full_pipeline.ipynb)  
Seules les colonnes `pdf_name`, `collectivity`, `year`, `competence` sont indispensables.  

Dans le répertoire `data_example/scraping/` :
- **Fichier `collectivity_list_for_scraping_example.csv` :**  
Example d'une liste de "petites" collectivités (1 seul service par compétence, 1 seule commune) pour lesquelles collecter des RPQS avec le module de scraping de NARVAL.


## Les données en sortie
Dans le répertoire `data_example/output/` :
- **Sous-répertoire `benchmark_*/` :**   
Exemple d'un répertoire typique créé en sortie du notebook [run_full_pipeline.ipynb](../notebooks/run_full_pipeline.ipynb). Contient les réponses de Narval pour chaque PDF dans `input/pdfs/`, à savoir:
  - des fichiers du type `RPQS_*_detailed_answers.csv` : réponses brutes de Narval, question par question, avant nettoyage
  - des fichiers du type `RPQS_*_answers.csv` : réponses nettoyées, avec au final une seule réponse par indicateur. Ces fichiers peuvent également être obtenus à partir des fichiers `RPQS_*_detailed_answers.csv` à l'aide du notebook [run_cleaning_step.ipynb](../notebooks/run_cleaning_step.ipynb)
  - des fichiers du type `RPQS_*_answers_vs_true.csv` : construits à partir des fichiers `RPQS_*_answers.csv` en ajoutant 2 colonnes contenant les "vraies" valeurs des indicateurs (contenues dans `input/sispea_vs_pdf_indic_values/v2/`) d'après la base SISPEA et d'après une annotation manuelle des PDFs. Uniquement nécessaire pour l'évaluation. Ces fichiers ne sont pas des outputs du notebook [run_full_pipeline.ipynb](../notebooks/run_full_pipeline.ipynb) mais du notebook [compute_metrics.ipynb](../notebooks/compute_metrics.ipynb).

- **Sous-répertoire `benchmark_table_*/` :**   
Exemple d'un répertoire typique créé en sortie du notebook [run_table_extraction_step.ipynb](../notebooks/run_table_extraction_step.ipynb). Même structure que `benchmark_*`.

- **Fichiers `all_metrics_per_pdf.csv` et `all_metrics_per_indic.csv` :**  
Exemples de fichiers contenant les métriques calculées par pdf et par indicateur, pour différents benchmarks.  
Notebook : [compute_metrics.ipynb](../notebooks/compute_metrics.ipynb) pour la création de ces fichiers à partir des résultats dans `output/benchmark_*`.

