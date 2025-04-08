# Description des notebooks

| Notebook  | Description |
| --- | ------------- |
| [tutorial.ipynb](./tutorial.ipynb)  | Tutoriel pour illustrer les différentes étapes de la pipeline Narval. |
| [run_full_pipeline.ipynb](./run_full_pipeline.ipynb)  | **Notebook principal permettant d'exécuter toute la pipeline Narval sur un jeu de PDFs.** Génère un répertoire du type `data/output/benchmark*` avec les réponses `RPQS_*_answers.csv` et `RPQS_*_detailed_answers.csv` à partir 1) des PDFs dans `data/input/pdfs/` listés dans `data/input/rpqs_eval_list_*.csv`, 2) d'un fichier de questions et mots-clés du type `data/input/question_keyword_*.csv` et 3) d'une liste des indicateurs SISPEA du type `data/input/indicateurs_*.csv`.   |
| [run_table_extraction_step.ipynb](./run_table_extraction_step.ipynb)  | Pour exécuter uniquement la pipeline partielle qui extrait les indicateurs à partir des tableaux récapitulatifs.|
| [run_cleaning_step.ipynb](./run_cleaning_step.ipynb)  | Pour exécuter uniquement la pipeline de nettoyage et de sélection des réponses brutes de Narval. Génère les fichiers du type `RPQS_*_answers.csv` à partir des `RPQS_*_detailed_answers.csv`.  |
| [compute_metrics.ipynb](./compute_metrics.ipynb)  | Pour calculer et écrire les métriques d'évaluation dans `data/output/all_metrics_per_pdf.csv` et `data/output/all_metrics_per_indic.csv`. Génère également les fichiers du type `RPQS_*_answers_vs_true.csv` |
| [plot_metrics.ipynb](./plot_metrics.ipynb)  | Pour tracer les métriques d'évaluation. |
| [rpqs_scraping.ipynb](./rpqs_scraping.ipynb)  | Pour scraper des RPQS à partir de SISPEA (fonctionnel) ou à partir de Google (non fonctionnel, module incomplet). Les PDFs scrapés sont sauvegardés respectivement dans `data/scraping/from_sispea/` et `data/scraping/from_google/`. Ils doivent être triés à la main.|
| [aws_local_file_transfer.ipynb](./aws_local_file_transfer.ipynb)  | Pour transférer des fichiers d'un bucket AWS S3 vers le répertoire local (ou inversement).  |
| [tests/test_*.ipynb](./tests/)  | Notebooks de tests des modules python de Narval. Ne sont pas forcément tous à jour.  |


