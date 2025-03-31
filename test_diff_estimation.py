import openai
import json
import re
import aiofiles
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from unidiff import PatchSet


def analyze_diff_path_set(patch_set):
    print_statements = []

    # Parse the hunk header to extract line numbers and changes
    for patched_file in patch_set:
        for hunk in patched_file:
            # Extract the line information from the hunk header
            old_start, old_lines = hunk.source_start, hunk.source_length
            new_start, new_lines = hunk.target_start, hunk.target_length
            print_statements.append(
                f"Changes start at line {old_start} in the original file and line {new_start} in the new file."
            )
            print_statements.append(
                f"{old_lines} lines were modified in the original file, and {new_lines} lines were modified in the new file."
            )

    # Count the number of added and removed lines
    added_lines = patch_set.added
    removed_lines = patch_set.removed

    print_statements.append(f"Total added lines: {added_lines}")
    print_statements.append(f"Total removed lines: {removed_lines}")

    return print_statements


def create_patch_set(patch):
    if not patch.startswith("---") or not patch.startswith("+++"):
        patch = f"--- a/file\n+++ b/file\n{patch}"

    # Create the PatchSet object
    patch_set = PatchSet.from_string(patch)
    return patch_set


def get_review_feedback(model, patch, language):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    diff_code = patch
    with open("prompt/diff_estimation_prompt.txt", "r") as prompt_file:
        prompt = prompt_file.read()
    cur_prompt = prompt.replace("{{diff_code}}", diff_code)

    patch = create_patch_set(diff_code)
    diff_code_information = analyze_diff_path_set(patch)
    diff_code_information = "\n".join(diff_code_information)
    cur_prompt = cur_prompt.replace("{{diff_code_information}}", diff_code_information)

    language_map = {
        "go": "Go",
        "java": "Java",
        "py": "Python",
        "cpp": "C++",
        "js": "JavaScript",
        "rb": "Ruby",
        "cs": "C#",
        "c": "C",
        "php": "PHP",
    }
    language = language_map.get(language, "")
    cur_prompt = cur_prompt.replace("{{language}}", language)

    # ChatGPT API에 질문하여 코드리뷰 필요 여부 판단
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=1,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    # bold markdown 제거
    answer = answer.replace("**", "")

    # Regular expression to extract the number
    match = (
        re.search(r"Code Review Required: \((\d)\)", answer)
        or re.search(r"Code Review Required: (\d)", answer)
        or re.search(r"Code Review Required (\d)", answer)
        or re.search(r"Code Review Required \((\d)\)", answer)
        or re.search(r"Code Review Required \(1-5\): (\d)", answer)
        or re.search(r"Code Review Required\): (\d)", answer)
        or re.search(r"Review Required: (\d)", answer)
        or re.search(r"Final Evaluation: (\d)", answer)
        or re.search(r"Final Evaluation: \((\d)\)", answer)
        or re.search(r"Final Evaluation \((\d\))", answer)
        or re.search(r"Final Evaluation \(1-5\): (\d)", answer)
        or re.search(r"Final Score: (\d)", answer)
        or re.search(r"Final Evaluation Score: (\d)", answer)
        or re.search(r"Final Evaluation \(overall\): (\d)", answer)
        or re.search(r"Final Evaluation \(1-5 scores ONLY\): (\d)", answer)
        or re.search(r"Code Review Required: Yes (\d)", answer)
        or re.search(r"Code Review Required: No (\d)", answer)
        or re.search(r"Final Evaluation \(Score: (\d)\):", answer)
        or re.search(
            r"Code Review Required \((\d\.\d)\)", answer
        )  # Decimal point support
    )

    # Extract and print the result if found
    score = 3
    if match:
        score = int(
            re.sub(r"[^\d]", "", match.group(1))
        )  # Remove non-numeric characters
    else:
        matches = re.findall(
            r"Score: (\d)|score: (\d)", answer
        )  # [('4', ''), ('4', ''), ('3', ''), ('4', ''), ('4', '')]
        if matches:
            # Extract the last non-empty match
            last_match = next(filter(None, matches[-1]))
            score = int(
                re.sub(r"[^\d]", "", last_match)
            )  # Remove non-numeric characters
            print(f"matches score found : {matches} -> {score}")
        else:
            print("!!!No score found!!!")
            print(answer)

    # Additional score parsing
    significance_match = re.search(r"Code Change Significance: (\d)", answer)
    complexity_match = re.search(r"Complexity of Changes: (\d)", answer)
    consistency_match = re.search(r"Code Consistency and Readability: (\d)", answer)
    risks_match = re.search(r"Potential Risks or Issues: (\d)", answer)

    significance = int(significance_match.group(1)) if significance_match else -1
    complexity = int(complexity_match.group(1)) if complexity_match else -1
    consistency = int(consistency_match.group(1)) if consistency_match else -1
    risks = int(risks_match.group(1)) if risks_match else -1

    return (1 if score >= 3 else 0), score, significance, complexity, consistency, risks


async def evaluate_patch(model, patch):
    y_true = int(patch["y"])
    language = patch.get("lang", "")
    review_needed, score, significance, complexity, consistency, risks = (
        await asyncio.to_thread(get_review_feedback, model, patch["patch"], language)
    )
    # await get_review_feedback(model, patch["patch"], language)

    y_pred = int(review_needed)

    patch["y_pred"] = review_needed
    patch["y_pred_score"] = score
    patch["y_code_change_significance"] = significance
    patch["y_complexity_of_changes"] = complexity
    patch["y_code_consistency_and_readability"] = consistency
    patch["y_potential_risks_or_issues"] = risks

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
    # model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    model = "o1-mini"

    input_file_name = "diff_estimation_sampling_100(seed1115).jsonl"
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
