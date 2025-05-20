import openai
import json
import re
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import asyncio
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def get_codereview(model, patch):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    #     prompt = """
    # Given this code “{{inputCode}}” and assuming you are an expert code reviewer:
    # 1) Decide whether the code needs to be
    #    revised or not. Answer True or False.
    # 2) If the response to the above point is True, then identify/classify the type
    #    of code change(s) required. Answer with
    #    one or more of the following:
    # • Changes are needed to refactor the code to improve its quality;
    # • Changes are needed since tests for this code must be written;
    # • Changes are needed to better align this code to good object-oriented design principles;
    # • Changes are needed to fix one or more bugs;
    # • Changes are needed to improve the logging of its execution;
    # • Changes are needed for other reasons not listed above.
    # 3) Write a code review (i.e., explain the changes to be performed, if any) based on your answers to the questions above.
    # """

    prompt = "Please review this code : {{inputCode}}"
    cur_prompt = prompt.replace("{{input_code}}", patch["patch"])

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=0.0,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    return answer


async def evaluate_patch(model, patch, TOP):

    code_review = await asyncio.to_thread(
        get_codereview, model, patch
    )  # IMPORTANT! get_classification 함수를 asyncio.to_thread로 비동기 실행

    patch["code_review"] = code_review

    return patch


def calculate_bleu(reference, hypothesis):
    # Smoothing function 선택 (method1, method2, or method4)
    smoothing_method = (
        SmoothingFunction().method5
    )  # You can change to method1, method2, or method4 here
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    bleu_score = sentence_bleu(
        reference,
        hypothesis,
        smoothing_function=smoothing_method,
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    return round(bleu_score, 4)  # 소수점 4자리까지 반올림


def calculate_bleu_ms(reference, hypothesis):
    from evaluator.smooth_bleu import bleu_fromstr

    reference = reference.split()
    hypothesis = hypothesis.split()
    return bleu_fromstr(hypothesis, reference, rmstop=False)


def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    # 일반적으로 사용하는 rouge1의 fmeasure 값을 반환
    rouge_score = scores["rougeL"].fmeasure
    return round(rouge_score, 4)  # 소수점 4자리까지 반올림


def save_metrics(patches, input_file_name, model, current_time):
    results = []
    for patch in patches:
        msg = patch.get("msg")
        code_review = patch.get("code_review")

        if msg:
            bleu_msg_code_review = calculate_bleu(msg, code_review)
            bleums_msg_code_review = calculate_bleu_ms(msg, code_review)
            rouge_msg_code_review = calculate_rouge(msg, code_review)
            patch["bleu_msg_code_review"] = bleu_msg_code_review
            patch["bleums_msg_code_review"] = bleums_msg_code_review
            patch["rouge_msg_code_review"] = rouge_msg_code_review

        results.append(patch)

    with open(
        f"output/1.3/{input_file_name}_{model}_vanila_{current_time}.jsonl", "a"
    ) as output_file:
        for result in results:
            output_file.write(json.dumps(result) + "\n")
    print(
        f"Results saved to output/1.3/{input_file_name}_{model}_vanila_{current_time}.jsonl"
    )


async def main():
    # 모델 설정
    model = "gpt-3.5-turbo"
    # model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    # model = "o1-mini"

    # input_file_name = "msg-test-1000.jsonl_gpt-3.5-turbo_20250311_022242.jsonl"
    input_file_name = "msg-test-1000.jsonl_gpt-4o-mini_20250311_020535.jsonl"

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
