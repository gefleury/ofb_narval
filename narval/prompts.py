NO_ANSWER_TAG = "je ne trouve pas"

T5_PROMPT_V1 = """
Contexte: {context} 
Question: {question} en {year} ?
"""

T5_PROMPT_V2 = """
Contexte: {context}
Réponds à la question suivante. La réponse est à donner {unit_tag} (s'il n'y a pas de réponse, réponds "{no_answer_tag}").
Question: {question} en {year} ?
"""

LLAMA_PROMPT_SYSTEM_V1 = """
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
Instructions : 
- La réponse est à donner {unit_tag}. 
- Si tu n'as pas assez d'information pour répondre, réponds "{no_answer_tag}".
-------
Extraits : {context}
"""

LLAMA_PROMPT_USER_V1 = """
Question : {question} en {year} ?
"""

LLAMA_PROMPT_SYSTEM_V2 = """
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
Instructions : 
- La réponse est à donner {unit_tag}. 
- Si tu n'as pas assez d'information pour répondre, réponds "{no_answer_tag}".
- Sois concis. La réponse doit être un nombre (dans les bonnes unités) ou "{no_answer_tag}".
-------
Extraits : {context}
"""

LLAMA_PROMPT_USER_V2 = """
Question : {question} en {year} ?
"""

LLAMA_PROMPT_SYSTEM_V3 = """
Extraits : {context}
"""

LLAMA_PROMPT_USER_V3 = """
Tu dois extraire la valeur d'un indicateur à partir de ces extraits issus d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
Instructions : 
- La réponse est à donner {unit_tag}. 
- Si tu n'as pas assez d'information pour répondre, réponds "{no_answer_tag}".
- Sois concis. La réponse doit être un nombre (dans les bonnes unités) ou "{no_answer_tag}".
-------
Question : {question} en {year} ?
"""

LLAMA_PROMPT_SYSTEM_V4 = """
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
"""

LLAMA_PROMPT_USER_V4 = """
# EXTRAITS #
{context}
#######
# QUESTION #
{question} en {year} ?
#######
# INSTRUCTIONS #
- La valeur de l'indicateur à trouver est un nombre exprimé {unit_tag}. {specific_instruction}
- Si tu ne trouves pas la réponse pour l'année {year} mais pour une autre année, réponds "{no_answer_tag}".
- Si tu n'as pas assez d'information pour répondre, réponds "{no_answer_tag}".
- Sois le plus concis possible. Donne uniquement la réponse : soit un nombre (dans les bonnes unités), soit "{no_answer_tag}".
"""

LLAMA_PROMPT_SYSTEM_V5 = """
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
Instructions : 
- La valeur de l'indicateur à trouver est un nombre exprimé {unit_tag}. {specific_instruction}
- Si tu ne trouves pas la réponse pour l'année {year} mais pour une autre année, réponds "{no_answer_tag}".
- Si tu n'as pas assez d'information pour répondre, réponds "{no_answer_tag}".
- Sois le plus concis possible. Donne uniquement la réponse : soit un nombre (dans les bonnes unités), soit "{no_answer_tag}".
-------
Extraits : {context}
"""

LLAMA_PROMPT_USER_V5 = """
Question : {question} en {year} ?
"""

LLAMA_PROMPT_SYSTEM_V6 = """
Tu es un assistant administratif qui répond à des questions sur les services d'{competence} en France.
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
-------
Extraits : {context}
"""

LLAMA_PROMPT_USER_V6 = """
Question : {question} en {year} ?
-------
Instructions : 
- La valeur de l'indicateur à trouver est un nombre exprimé {unit_tag}. {specific_instruction}
- Si tu ne trouves pas la réponse pour l'année {year} dans l'extrait, réponds "{no_answer_tag}".
- Si tu n'as pas assez d'information dans l'extrait pour répondre, réponds "{no_answer_tag}".
- Sois le plus concis possible. Ta réponse doit être uniquement un nombre (dans les bonnes unités) ou "{no_answer_tag}".
"""

LLAMA_PROMPT_SYSTEM_V7 = """
Tu es un assistant administratif qui répond à des questions sur les services d'{competence} en France.
Tu dois extraire la valeur d'un indicateur à partir d'extraits d'un rapport sur l'{competence} en {year} dans la collectivité "{collectivity}".
-------
Instructions : 
- La valeur de l'indicateur à trouver est un nombre exprimé {unit_tag}. {specific_instruction}
- Si tu ne trouves pas la réponse pour l'année {year} dans l'extrait, réponds "{no_answer_tag}".
- Si tu n'as pas assez d'information dans l'extrait pour répondre, réponds "{no_answer_tag}".
- Sois le plus concis possible. Ta réponse doit être uniquement un nombre (dans les bonnes unités) ou "{no_answer_tag}".
-------
Extraits : {context}
"""

LLAMA_PROMPT_USER_V7 = """
Question : {question} en {year} ?
"""

# Dictionary to store various prompts
prompts_dict = {
    "T5_prompt_v1": T5_PROMPT_V1,
    "T5_prompt_v2": T5_PROMPT_V2,
    "Llama_prompt_system_v1": LLAMA_PROMPT_SYSTEM_V1,
    "Llama_prompt_user_v1": LLAMA_PROMPT_USER_V1,
    "Llama_prompt_system_v2": LLAMA_PROMPT_SYSTEM_V2,
    "Llama_prompt_user_v2": LLAMA_PROMPT_USER_V2,
    "Llama_prompt_system_v3": LLAMA_PROMPT_SYSTEM_V3,
    "Llama_prompt_user_v3": LLAMA_PROMPT_USER_V3,
    "Llama_prompt_system_v4": LLAMA_PROMPT_SYSTEM_V4,
    "Llama_prompt_user_v4": LLAMA_PROMPT_USER_V4,
    "Llama_prompt_system_v5": LLAMA_PROMPT_SYSTEM_V5,
    "Llama_prompt_user_v5": LLAMA_PROMPT_USER_V5,
    "Llama_prompt_system_v6": LLAMA_PROMPT_SYSTEM_V6,
    "Llama_prompt_user_v6": LLAMA_PROMPT_USER_V6,
    "Llama_prompt_system_v7": LLAMA_PROMPT_SYSTEM_V7,
    "Llama_prompt_user_v7": LLAMA_PROMPT_USER_V7,
}


def get_prompt(input_dict: dict, version="T5_prompt_v1"):
    """
    Retrieve and format a prompt

    Args:
    - version (str): prompt version.
    - input_dict (dict):
        Dictionary containing values to replace the placeholders (context, question, unit, ...)

    Returns:
    - str: The formatted prompt
    """
    try:
        prompt = prompts_dict[version]
    except KeyError as e:
        raise KeyError(f"The prompt version {version} does not exist") from e
    try:
        prompt = prompt.format(no_answer_tag=NO_ANSWER_TAG, **input_dict)
    except KeyError as e:
        raise ValueError(f"Missing key in input_dict for placeholder {e}") from e
    return prompt
