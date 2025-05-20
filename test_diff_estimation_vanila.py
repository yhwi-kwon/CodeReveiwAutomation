import openai
import json
import re
import aiofiles
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_review_feedback(model, patch):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    diff_code = patch
    prompt = "Please review whether the code change needs a review.  : {{diff_code}}"
    prompt = prompt + "\n0 : Code Review Not Required \n1 : Code Review Required"
    prompt = prompt + "Expected Output : 0 or 1"
    cur_prompt = prompt.replace("{{diff_code}}", diff_code)

    # ChatGPT API에 질문하여 코드리뷰 필요 여부 판단
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=0.0,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    # bold markdown 제거
    answer = answer.replace("**", "")

    # Regular expression to extract the number
    match = (
        re.search(r"(\d)", answer)
        or re.search(r"Code Review Required: (\d)", answer)
        or re.search(r"Code Review Not Required: (\d)", answer)
    )

    score = 1
    # print(answer)
    if match:
        score = int(match.group(1))
    else:
        print(answer)
    # print(score)
    return score


async def evaluate_patch(model, patch):
    y_true = int(patch["y"])
    review_needed = await asyncio.to_thread(get_review_feedback, model, patch["patch"])
    # await get_review_feedback(model, patch["patch"], language)

    y_pred = int(review_needed)
    patch["y_pred"] = review_needed

    return y_true, y_pred, patch


async def save_metrics(y_true_list, y_pred_list, input_file_name, model, current_time):
    precision = precision_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list)
    f1 = f1_score(y_true_list, y_pred_list)
    accuracy = accuracy_score(y_true_list, y_pred_list)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    result = {
        "current_time": current_time,
        "input_file_name": input_file_name,
        "model": model,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy,
    }

    async with aiofiles.open(
        f"output/1.1/{input_file_name}_{model}_{current_time}.result", "w"
    ) as result_file:
        await result_file.write(json.dumps(result, indent=4))
    print(
        f"Metrics saved to output/1.1/{input_file_name}_{model}_{current_time}.result"
    )


async def main():
    # 모델 설정
    # model = "gpt-3.5-turbo"
    model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    # model = "o1-mini"

    input_file_name = "diff_estimation_sampling_1000(seed1115).jsonl"
    # input_file_name = "cls-test.jsonl"

    async with aiofiles.open(f"data/{input_file_name}", "r") as file:
        patches = [json.loads(line) for line in await file.readlines()]

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = f"output/1.1/{input_file_name}_{model}_{current_time}.jsonl"

    async def process_patch(patch, model, pbar):
        """단일 패치를 평가하고 결과를 반환"""
        try:
            y_true, y_pred, updated_patch = await evaluate_patch(model, patch)
            pbar.update(1)  # tqdm 수동 업데이트
            return y_true, y_pred, updated_patch
        except Exception as e:
            print(f"Error processing patch: {e}")
            return None

    # tqdm 설정
    progress_bar = tqdm_asyncio(
        total=len(patches), desc="Processing patches", dynamic_ncols=True
    )

    # 비동기 작업 생성
    tasks = [
        asyncio.create_task(process_patch(patch, model, progress_bar))
        for patch in patches
    ]

    # 모든 패치 병렬 처리
    results = await asyncio.gather(*tasks)

    progress_bar.close()  # tqdm 닫기

    # 결과 저장 (동기 방식으로 처리하여 속도 최적화)
    with open(output_path, "w") as output_file:
        y_true_list = []
        y_pred_list = []

        for result in results:
            if result:
                y_true, y_pred, updated_patch = result
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
                output_file.write(json.dumps(updated_patch) + "\n")

    # 성능 지표 저장
    await save_metrics(y_true_list, y_pred_list, input_file_name, model, current_time)


if __name__ == "__main__":
    asyncio.run(main())
