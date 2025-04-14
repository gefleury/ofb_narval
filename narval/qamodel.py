import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

from narval.prompts import NO_ANSWER_TAG, get_prompt


class T5QuestionAnswering:
    def __init__(self, model_name="google/flan-t5-xl"):
        assert model_name in [
            "google/flan-t5-base",
            "google/flan-t5-small",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ]

        # Initialize the model and tokenizer
        # General
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # For T5 models
        # Use device_map="auto" to automatically map model to multiple GPUs or CPU if needed
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        )

        print("Device = ", self.model.device)

    def format_prompt(
        self,
        prompt_params: dict,
        prompt_version="T5_prompt_v1",
    ):
        prompt = get_prompt(prompt_params, version=prompt_version)

        return prompt

    def predict(
        self,
        prompt: str,
        tokenizer_max_length=512,  # Max length for the entire input (context+question)
        tokenizer_truncation="only_second",
        tokenizer_padding="max_length",
        max_new_tokens=10,  # Short answers are expected
        num_beams=3,  # Increase num_beams for better results (but will increase computation time)
        early_stopping=True,  # Stop generating once a complete answer is found
    ):
        """
        Generates an answer for a given prompt

        Parameters
        ----------
        prompt: str
        tokenizer_max_length: Max number of tokens for the entire input ie context+question (int)
        tokenizer_truncation:
        tokenizer_padding:
        max_new_tokens: Max number of tokens for the generated answer (int)
        model_num_beams: Number of beams used in beam search (int)
        model_early_stopping: bool

        Returns
        -------
        The predicted answer (str)
        """
        # Tokenize the input and move tensors to the appropriate device
        inputs = self.tokenizer(
            prompt,
            max_length=tokenizer_max_length,  # Max length for the entire input (context+question)
            truncation=tokenizer_truncation,
            padding=tokenizer_padding,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.model.device)

        # Print the number of input tokens without truncation (for checking)
        # check_inputs = self.tokenizer(
        #     prompt,
        #     truncation=False,
        #     return_tensors="pt",
        # )
        # num_tokens = len(check_inputs["input_ids"][0])
        # git print("Number of input tokens = ", num_tokens)

        # Generate output using the model
        outputs = self.model.generate(
            input_ids,
            max_length=max_new_tokens,  # Short answers are expected
            num_beams=num_beams,  # Increase num_beams for better results (but will increase computation time)
            early_stopping=early_stopping,  # Stop generating once a complete answer is found
        )

        # Decode the output back to a string
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer == "":
            answer = NO_ANSWER_TAG

        return answer


class Llama3QuestionAnswering:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        assert model_name in [
            "meta-llama/Meta-Llama-3-8B-Instruct",  # Add more if appropriate
        ]

        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    # "quantization_config": {"load_in_4bit": True},     # raises an error
                    # "low_cpu_mem_usage": True,  # apparently slows down the calculation (to be checked!!)
                },
                device_map="auto",
            )
            print("Device = ", self.pipe.model.device)
        except AttributeError as e:
            raise AttributeError(
                """ 
                Consider login to HuggingFace Hub first.
                Create your HF_TOKEN on your HuggingFace profile, 
                save it as an environment variable and then run
                from huggingface_hub import login
                hf_token = os.environ["HF_TOKEN"]
                login(token = hf_token)
                """
            ) from e

    def format_prompt(
        self,
        prompt_params: dict,
        prompt_version=("Llama_prompt_system_v1", "Llama_prompt_user_v1"),
    ):
        system_content = get_prompt(prompt_params, version=prompt_version[0])
        user_content = get_prompt(prompt_params, version=prompt_version[1])

        return system_content, user_content

    def predict(
        self,
        prompt,
        max_new_tokens=50,
    ):
        """
        Generates an answer for a given prompt.

        Parameters
        ----------
        prompt: a tuple giving the system and user prompt
        max_new_tokens: Max number of tokens for the generated answer (int)

        Returns
        -------
        The predicted answer (str)
        """
        system_content, user_content = prompt

        inputs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipe(
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            # To remove the warning "Setting `pad_token_id` to `eos_token_id`:None for open-end generation."
            # Not yet checked. There might be better choices
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            # For reproducibility
            do_sample=False,
            top_p=None,
            temperature=None,
        )

        answer = outputs[0]["generated_text"][-1]["content"]

        if answer == "":
            answer = NO_ANSWER_TAG

        return answer


class QAModel:
    def __init__(self, model_name="google/flan-t5-xl"):
        """
        Initialize the QAModel with either a T5Model or LlamaModel
        based on the provided model_name.

        Args:
            model_name (str): The name of the model to instantiate. Must be either 'T5' or 'Llama3'.
        """
        if model_name in [
            "google/flan-t5-base",
            "google/flan-t5-small",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ]:
            self.model = T5QuestionAnswering(model_name=model_name)
        elif model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
            self.model = Llama3QuestionAnswering(model_name=model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}. ")

    def format_prompt(
        self,
        prompt_params: dict,
        prompt_version,
    ):
        return self.model.format_prompt(prompt_params, prompt_version)

    def predict(self, prompt: str, **kwargs):
        return self.model.predict(prompt, **kwargs)
