import os
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


def get_few_shots(path, dataset="corpus"):
    train = pd.read_csv(path)
    few_shots = (
        train.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 3), random_state=313))
        .sample(frac=1, random_state=313)
        .reset_index(drop=True)
    )
    few_shots.to_csv(f"data/cache/few_shots_{dataset}.csv")


def prompt_wrapper(few_shots):
    def apply_prompt(inp):
        prompt = ""

        for _, row in few_shots.iterrows():
            prompt += f"{row['sentence']}####<{row['label']}><|endoftext|>\n"

        prompt += f"{inp['sentence']}####"

        return {"prompt": prompt}

    return apply_prompt


def prompt_corpus():
    try:
        few_shots = pd.read_csv("data/cache/few_shots_corpus.csv")
    except Exception:
        print("Could not find file, please run prepare first")
        exit(1)

    all_data = load_dataset(
        "csv",
        data_files={
            "train": ["data/corpus_train.csv"],
            "validation": ["data/corpus_valid.csv"],
            "test": ["data/corpus_test.csv"],
        },
    )
    return all_data["validation"].map(prompt_wrapper(few_shots))


def prompt_cmv():
    few_shots = pd.read_csv("data/cache/few_shots_cmv.csv")

    validation = pd.read_csv("data/cmv_train.csv")

    for i, row in validation.iterrows():
        apply_prompt = prompt_wrapper(few_shots)
        yield i, apply_prompt(row)["prompt"]


def load_gpt_j():
    cache_dir = os.getenv("CACHE_DIR", "NOT SET")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B", cache_dir=cache_dir
    )

    return model, tokenizer


def prompt(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.4,
        max_length=input_ids.size()[1] + 6,
    )

    return tokenizer.batch_decode(gen_tokens)[0]
