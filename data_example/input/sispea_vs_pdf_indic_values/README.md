## Description des répertoires dans `data/input/sispea_vs_pdf_indic_values`
- **Répertoire `v1`: 1ère version.**   
    - PDFs de `rpqs_eval_list_1.csv` : annotés en utilisant les valeurs saisies dans SISPEA; 
    - PDFs de `rpqs_eval_list_2.csv` : annotés sans connaître les valeurs saisies ou non dans SISPEA.  
- **Répertoire `v2`: 2ème version.**   
On a revu la labélisation après analyse des erreurs :
    - PDFs de `rpqs_eval_list_1.csv` : on décide de les annoter sans connaître les valeurs saisies ou non dans SISPEA. Typiquement, les valeurs qui étaient saisies dans SISPEA sans être explicitement dans le PDF mais peut-être recalculables avec les infos du PDF sont maintenant mises à `NaN` pour être plus cohérent avec la labélisation des PDFs de `rpqs_eval_list_2.csv`
    - PDFs de `rpqs_eval_list_2.csv` : comme dans le `v1` avec correction de la labélisation pour P256.2 $\rightarrow$ quand l'encours de la dette est indiqué comme étant de 0€, on labélise la durée d'extinction de la dette à 0 année. 

### Règles de labélisation dans `v2`
- Pour les indicateurs de conformité P203.3, P204.3, P205.3: si "non conforme" alors "0", si "conforme" alors "100"
- Pour l'indicateur de conformité P254.3: si "non conforme" alors `NaN`, si "conforme" alors "100"
- Pour P256.2: 
    - Si l'encours de la dette est explicitement indiquée comme étant de 0€ alors on labélise la durée d'extinction de la dette à 0 année. 
    - Si la dette est marquée comme remboursée en "2036" par exemple alors que le rapport est pour l'année "2020" alors on labélise à "16". 
    - Pas plus de calculs que ça. Si la durée d'extinction n'est pas indiquée explicitement alors on met `NaN`
- Pour D203.0: on fait la somme des quantités de boues en tMS pour les différentes stations si le total n’est pas donné explicitement 
- Pour D204.0 :
    - Si le prix TTC est mentionné pour 120m3 mais pas en €/m3 alors on fait la division par 120 pour l'avoir en €/m3
    - Pas plus de calculs que ça. Si le prix n'est pas indiqué explicitement alors on met `NaN`
- De façon générale, si un indicateur est marqué "NEANT" alors on met `NaN`
- De façon générale, si la valeur n'est pas indiquée explicitement alors on met `NaN`
- Da façon générale, si la valeur d'un indicateur n'est pas donnée pour l'année de l'exercice courant mais pour une autre année alors on met `NaN` (notamment pour D204.0, la valeur de l'année N est celle au 1er janvier de l'année N+1)
- S'il y a une contradiction entre le(s) tableau(x) et le texte, on garde la valeur du tableau, sauf si cette valeur est manifestement absurde (coquille, oubli de convertir le prix en €/m3, …) et dans ce cas on prend la valeur dans le texte
- Des erreurs humaines de labélisation restent possibles (sur les 855 valeurs ...). Dans certains cas, la bonne labélisation n'est pas claire et il est fort probable que d'autres annotateurs fourniraient des labélisations différentes.

