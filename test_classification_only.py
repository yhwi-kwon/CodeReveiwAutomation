import openai
import json
import re
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import asyncio


def get_classification(model, input_code):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    prompt = open("prompt/classification_prompt.txt").read()
    cur_prompt = prompt.replace("{{input_code}}", input_code)
    cur_prompt = prompt.replace("{{diff_code}}", input_code)

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

    # Extract categories, subcategories, and reasons
    primary_category = extract_info(r"Primary Category: (.+)", answer, to_lower=True)
    primary_subcategory = extract_info(r"Primary Subcategory: (.+)", answer)
    primary_reason = extract_info(r"Primary Reason: (.+)", answer)

    secondary_category = extract_info(
        r"Secondary Category: (.+)", answer, to_lower=True
    )
    secondary_subcategory = extract_info(r"Secondary Subcategory: (.+)", answer)
    secondary_reason = extract_info(r"Secondary Reason: (.+)", answer)

    tertiary_category = extract_info(r"Tertiary Category: (.+)", answer, to_lower=True)
    tertiary_subcategory = extract_info(r"Tertiary Subcategory: (.+)", answer)
    tertiary_reason = extract_info(r"Tertiary Reason: (.+)", answer)

    return (
        primary_category,
        primary_subcategory,
        primary_reason,
        secondary_category,
        secondary_subcategory,
        secondary_reason,
        tertiary_category,
        tertiary_subcategory,
        tertiary_reason,
    )


async def evaluate_patch(model, patch, TOP):
    input_code = patch["patch"]
    (
        primary_category,
        primary_subcategory,
        primary_reason,
        secondary_category,
        secondary_subcategory,
        secondary_reason,
        tertiary_category,
        tertiary_subcategory,
        tertiary_reason,
    ) = await asyncio.to_thread(
        get_classification, model, input_code
    )  # IMPORTANT! get_classification 함수를 asyncio.to_thread로 비동기 실행

    primary_category = primary_category.strip().lower() or "other"
    secondary_category = secondary_category.strip().lower() or "other"
    tertiary_category = tertiary_category.strip().lower() or "other"

    patch["primary_category"] = primary_category
    patch["primary_subcategory"] = primary_subcategory
    patch["primary_reason"] = primary_reason
    patch["secondary_category"] = secondary_category
    patch["secondary_subcategory"] = secondary_subcategory
    patch["secondary_reason"] = secondary_reason
    patch["tertiary_category"] = tertiary_category
    patch["tertiary_subcategory"] = tertiary_subcategory
    patch["tertiary_reason"] = tertiary_reason

    # Check if primary_category, secondary_category, or tertiary_category is in y_true categories
    return patch


def save_metrics(input_file_name, model, current_time, TOP):
    result = {
        "current_time": current_time,
        "input_file_name": input_file_name,
        "model": model,
        "TOP": TOP,
    }

    with open(
        f"output/1.2/{input_file_name}_{model}_{current_time}.result", "a"
    ) as result_file:
        json.dump(result, result_file, indent=4)
        result_file.write("\n")
    print(
        f"Metrics saved to output/1.2/{input_file_name}_{model}_{current_time}.result"
    )


async def main():
    # 모델 설정
    model = "gpt-3.5-turbo"
    # model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    # model = "o1-mini"

    TOP = 3

    # input_file_name = "df_clnl_4.jsonl"
    input_file_name = "msg-test-1000.jsonl"

    with open(f"data/{input_file_name}", "r") as file:
        patches = [json.loads(line) for line in file]

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def process_top(top):
        y_true_list = []
        y_pred_list = []
        tasks = []

        pbar = tqdm_asyncio(total=len(patches), desc=f"Processing TOP={top}")

        async def process_patch(patch):
            result = await evaluate_patch(model, patch, top)
            pbar.update(1)  # tqdm_asyncio를 올바르게 업데이트
            return result

        tasks = [asyncio.create_task(process_patch(patch)) for patch in patches]
        results = await asyncio.gather(*tasks)

        pbar.close()  # Progress bar 닫기

        with open(
            f"output/1.2/{input_file_name}_{model}_{current_time}.jsonl", "a"
        ) as output_file:
            for updated_patch in results:
                output_file.write(json.dumps(updated_patch) + "\n")

        save_metrics(input_file_name, model, current_time, top)

    # 병렬 실행을 보장하기 위해 asyncio.create_task를 사용한 후 gather 실행
    tasks = [asyncio.create_task(process_top(top)) for top in range(3, TOP + 1)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
