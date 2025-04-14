# Narval

Narval est un projet développé au sein de l'Unité Données et Appui Méthodologique de l'Office Français de la Biodiversité (OFB), en collaboration avec le Service Eaux et Milieux Aquatiques de l'OFB.

## Description du projet
*Pour une description complète du projet, voir ce [rapport (PDF)](rapport_narval.pdf).*  

Narval permet d'extraire les valeurs d'indicateurs [SISPEA](https://services.eaufrance.fr/) à partir de Rapports sur le Prix et la Qualité des Services d'eau et d'assainissement (RPQS) au format PDF. Etant donné un rapport PDF en entrée, Narval fournit en sortie un tableau des indicateurs au format csv.

A ce jour, Narval traite uniquement :
- les rapports PDFs non scannés
- les rapports PDFs relatifs à l'assainissement collectif (19 indicateurs de cette [liste](https://services.eaufrance.fr/indicateurs) (hors P202.2A))
- les rapports PDFs de "petites" collectivités i.e. constituées d'une seule entité de gestion d'assainissement collectif et d'une seule commune

La performance de Narval peut être évaluée (entre autres) par la métrique d'*accuracy* c'est-à-dire 
le pourcentage de bonnes réponses extraites (en comparaison des vraies valeurs des indicateurs 
écrites dans le PDF). Actuellement, sur un jeu test de 45 RPQS, Narval présente une *accuracy*
d'environ 90% (tous indicateurs confondus). 

## Organisation du repo
```
- narval/       : le code Python
- notebooks/    : notebooks pour lancer le code et faire des analyses 
- data_example/ : exemple de données (à copier-coller et renommer en data/)
```

## Installation
Narval a été développé et testé dans un environnement Python 3.12.

### Installation en local (Ubuntu)
 
- Utiliser par exemple `pyenv` pour installer Python 3.12 
- Clôner le repo `narval`
- Dans le repo `narval`, créer un environnement virtuel avec par exemple `python -m venv .venv` 
- Activer l'environnement virtuel avec `source .venv/bin/activate`
- Installer les packages requis avec
`pip install -r requirements.txt`

### Installation en local (Windows)
Utiliser Anaconda ou installer WSL2 puis suivre les instructions d'installation sur Ubuntu. 

### Installation sur le SSP Cloud
Uniquement en cas de compte disponible sur le [SSP Cloud](https://datalab.sspcloud.fr).

- Connecter le SSP Cloud à Git :
    - créer un access token sur Gitlab (dans `User settings`, onglet `Access tokens`)
    - sur le SSP Cloud, rentrer ce token dans `Mon Compte`, onglet `Git`
    - sur le SSP Cloud, avant d'ouvrir un service VSCode ou Jupyter Notebook, 
    aller dans le panneau de configuration du service et dans l'onglet `Git`, ajouter l'url du repo `https://gitlab.ofb.fr/genevieve.fleury/narval.git`
- Veiller à avoir préalablement ajouté les variables d'environnement (voir section suivante)
- Lancer un service `Vscode-pytorch-gpu` ou `Jupyter-pytorch-gpu` puis ouvrir un terminal
- Aller dans le repo `narval` puis installer les packages requis avec
`pip install -r requirements.txt`.  
L'erreur liée à `prompt-toolkit` peut être ignorée.


## Variables d'environnement

### En local
Dans un fichier `.env` à la racine du repo, inclure les variables d'environnements suivantes (selon le besoin):
- `HF_TOKEN` : pour l'accès à Hugging Face (token à créer à partir d'un compte [Hugging Face](https://huggingface.co/))
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, `S3_BUCKET_FROM_LOCAL`: pour accéder au stockage AWS S3 mis à disposition par le [SSP Cloud](https://datalab.sspcloud.fr) à partir d'une machine extérieure au service du datalab. Ces 4 premières variables sont disponibles à partir d'un compte SSP Cloud, en allant dans l'onglet `Mon Compte` puis `Connexion au stockage`. La variable `S3_BUCKET_FROM_LOCAL` est l'identifiant du bucket d'un utilisateur (du type "1ère lettre du prénom + nom" de l'utilisateur). Il est accessible à partir d'un compte [SSP Cloud](https://datalab.sspcloud.fr) en allant dans l'onglet `Mes fichiers`. Toutes ces variables sont uniquement nécessaires pour transférer des fichiers d'un répertoire du dépôt S3 vers un répertoire en local via le notebook `aws_local_file_transfer.ipynb`.


### Sur le SSP Cloud

Se connecter au [SSP Cloud](https://datalab.sspcloud.fr) et à partir de l'onglet `Mes secrets`, créer un nouveau secret `narval`. Y ajouter les 2 variables d'environnement suivantes : 
- `HF_TOKEN` : pour l'accès à Hugging Face (token à créer à partir d'un compte [Hugging Face](https://huggingface.co/))
- `S3_BUCKET` : identifiant du bucket de l'utilisateur (du type "1ère lettre du prénom + nom" de l'utilisateur). Accessible à partir de l'onglet `Mes fichiers`. 

Avant de lancer un service VSCode ou Jupyter Notebook, aller dans le panneau de configuration du service et dans l'onglet `Vault`, entrer `narval` comme secret.

## Données

Le répertoire [`data_example`](data_example/) fournit la structures des données à utiliser pour le projet Narval et des exemples de rapports PDFs à analyser (collectés à partir du [site SISPEA](https://www.services.eaufrance.fr/)).   


### En local
Copier-coller le répertoire `data_example` et le renommer en `data`. L'enrichir au besoin avec de nouvelles données.

### Sur le SSP Cloud
A partir d'un compte [SSP Cloud](https://datalab.sspcloud.fr), aller dans l'onglet `Mes fichiers` et créer un répertoire `narval`. Y copier-coller le répertoire `data_example` après l'avoir renommé en `data`, avec l'aide par exemple du notebook [aws_local_file_transfer.ipynb](notebooks/aws_local_file_transfer.ipynb). L'enrichir au besoin avec de nouvelles données.

## Usage
Narval utilise dans sa pipeline un modèle [Llama3-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) de taille 16Go, téléchargé à partir d'[Hugging Face](https://huggingface.co/) (sa licence doit être validée). Pour garder des temps de calcul raisonnables, il est fortement recommandé d'exécuter la pipeline complète sur une machine disposant d'un GPU $\gtrsim$ 16Go. En revanche, le développement du code et l'analyse des résultats peut être fait en local sur un PC de bureau.

### En local
Privilégier un travail en local pour :
- le calcul et l'analyse des métriques avec les notebooks [compute_metrics.ipynb](notebooks/compute_metrics.ipynb) et [plot_metrics.ipynb](notebooks/plot_metrics.ipynb)
- l'exécution de la pipeline partielle qui extrait les indicateurs des tableaux récapitulatifs, avec le notebook [run_table_extraction_step.ipynb](notebooks/run_table_extraction_step.ipynb)
- la modification des étapes de nettoyage des réponses de Narval, avec le notebook [run_cleaning_step.ipynb](notebooks/run_cleaning_step.ipynb)
- le développement du code

### Sur le SSP Cloud
Privilégier un travail sur le [SSP Cloud](https://datalab.sspcloud.fr) (ou autre machine de calcul avec GPU $\gtrsim$ 16Go) pour exécuter la pipeline complète d'extraction d'indicateurs SISPEA à partir de PDFs. Pour cela :
- se connecter au [SSP Cloud](https://datalab.sspcloud.fr) et lancer un service `Vscode-pytorch-gpu` 
ou `Jupyter-pytorch-gpu` en allant préalablement dans l'onglet `Ressources` du panneau de configuration 
du service pour demander l'accès à un 16-32Go $\lesssim$ GPU $\lesssim$ 64Go.
- Ouvrir un terminal et installer les packages requis comme expliqué plus haut. Vérifier également que les variables d'environnement sont mentionnées et que les données sont bien dans le répertoire `narval/data` sur le bucket S3.
- Exécuter le notebook [run_full_pipeline.ipynb](notebooks/run_full_pipeline.ipynb) (ou [tutorial.ipynb](notebooks/tutorial.ipynb)). Des données sont créées en sortie sur le bucket S3.
- En local, rapatrier ces nouvelles données à l'aide du notebook [aws_local_file_transfer.ipynb](notebooks/aws_local_file_transfer.ipynb)



## Remerciements
Le projet Narval a bénéficié des ressources suivantes :
- [SSP Cloud](https://datalab.sspcloud.fr) : pour la mise à disposition de moyens de calcul et d'un espace de stockage
- [Hugging Face](https://huggingface.co/) : pour l'accès aux modèles de langage [Flan-T5](https://huggingface.co/google/flan-t5-xl) (Google) et [Llama3-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (Meta)
- [PyPDF2](https://pypdf2.readthedocs.io) : pour l'extraction de texte à partir de PDFs
- [PDFPlumber](https://github.com/jsvine/pdfplumber) : pour l'extraction de tableaux sous forme de pandas dataframes à partir de PDFs
- [Pandas](https://pandas.pydata.org/) : pour la manipulation de données tabulaires
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) : pour le scraping de PDFs
- [CodeCarbon](https://github.com/mlco2/codecarbon) : pour l'estimation de l'impact carbone de Narval
- l'écosystème Python
- [Narval V0](https://github.com/malouberthe/Narval) : la première version de Narval 

## Licence
Le code de NARVAL est publié sous [licence MIT](licence_MIT.md).  
Les données sont publiées sous [Etalab Licence Ouverte 2.0](licence_Etalab2.md).  
Les modèles de langage Flan-t5 et Llama3 utilisés dans NARVAL sont distribués respectivement sous licence [Apache2.0](https://choosealicense.com/licenses/apache-2.0/) et [Llama3](https://www.llama.com/llama3/license/). 

