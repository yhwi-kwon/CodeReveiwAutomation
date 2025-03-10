import openai
import json
import re
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import asyncio
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def get_classification(model, input_code, primary_category):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    # Load the appropriate prompt based on the primary category
    prompt_file = f"prompt/{primary_category.replace(' ', '_').lower()}.txt"
    prompt = open(prompt_file).read()
    cur_prompt = prompt.replace("{{input_code}}", input_code)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=0.0,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    # bold markdown 제거
    answer = answer.replace("**", "")

    def extract_info(pattern, text, default="unknown", to_lower=False):
        match = re.search(pattern, text)
        if match:
            result = match.group(1).strip()
            return result.lower() if to_lower else result
        return default

    # Extract Type, Reasoning, and Code Review
    review_type = extract_info(r"Type: (.+)", answer)
    reasoning = extract_info(r"Reasoning: (.+)", answer)
    code_review = extract_info(r"Code Review: (.+)", answer)

    return review_type, reasoning, code_review


async def evaluate_patch(model, patch, TOP):
    input_code = patch["input_code"]
    primary_category = patch["primary_category"]

    review_type, reasoning, code_review = await asyncio.to_thread(
        get_classification, model, input_code, primary_category
    )  # IMPORTANT! get_classification 함수를 asyncio.to_thread로 비동기 실행

    patch["review_type"] = review_type
    patch["reasoning"] = reasoning
    patch["code_review"] = code_review

    return patch


def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    bleu_score = sentence_bleu(reference, hypothesis)
    return round(bleu_score, 4)  # 소수점 4자리까지 반올림


def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    # 일반적으로 사용하는 rouge1의 fmeasure 값을 반환
    rouge_score = scores["rougeL"].fmeasure
    return round(rouge_score, 4)  # 소수점 4자리까지 반올림


def save_metrics(patches, input_file_name, model, current_time):
    results = []
    for patch in patches:
        target = patch.get("target", "")
        code_review = patch.get("code_review", "")
        output = patch.get("output", "")

        bleu_target_code_review = calculate_bleu(target, code_review)
        rouge_target_code_review = calculate_rouge(target, code_review)

        bleu_target_output = calculate_bleu(target, output)
        rouge_target_output = calculate_rouge(target, output)

        patch["bleu_code_review"] = bleu_target_code_review
        patch["rouge_code_review"] = rouge_target_code_review
        patch["bleu_output"] = bleu_target_output
        patch["rouge_output"] = rouge_target_output

        results.append(patch)

    with open(
        f"output/1.3/{input_file_name}_{model}_{current_time}.jsonl", "a"
    ) as output_file:
        for result in results:
            output_file.write(json.dumps(result) + "\n")
    print(f"Results saved to output/1.3/{input_file_name}_{model}_{current_time}.jsonl")


async def main():
    # 모델 설정
    # model = "gpt-3.5-turbo"
    model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    # model = "o1-mini"

    input_file_name = "df_clnl_4_test3.jsonl_gpt-4o-mini_20250305_020130.jsonl"

    with open(f"data/{input_file_name}", "r") as file:
        patches = [json.loads(line) for line in file]

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def process_top(top):
        tasks = []

        pbar = tqdm_asyncio(total=len(patches), desc=f"Processing TOP={top}")

        async def process_patch(patch):
            result = await evaluate_patch(model, patch, top)
            pbar.update(1)  # tqdm_asyncio를 올바르게 업데이트
            return result

        tasks = [asyncio.create_task(process_patch(patch)) for patch in patches]
        results = await asyncio.gather(*tasks)

        pbar.close()  # Progress bar 닫기

        save_metrics(results, input_file_name, model, current_time)

    # 병렬 실행을 보장하기 위해 asyncio.create_task를 사용한 후 gather 실행
    tasks = [asyncio.create_task(process_top(top)) for top in range(1, 2)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
