import openai
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import asyncio


def get_codereview(model, patch):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    prompt_file = f"prompt/generate_codereview_prompt.txt"
    prompt = open(prompt_file).read()

    input_code = patch.get("patch") or patch.get("input_code")

    cur_prompt = prompt.replace("{{input_code}}", input_code)

    cur_prompt = cur_prompt.replace("{{primary_category}}", patch["primary_category"])
    cur_prompt = cur_prompt.replace(
        "{{primary_subcategory}}", patch["primary_subcategory"]
    )
    cur_prompt = cur_prompt.replace("{{primary_reason}}", patch["primary_reason"])

    if patch.get("lang"):
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
        language = language_map.get(patch["lang"], "")
        cur_prompt = cur_prompt.replace("{{language}}", language)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=0.0,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    # bold markdown 제거
    answer = answer.replace("**", "")

    return answer


def get_vanila_gpt_codereview(model, patch):
    # OpenAI API 키 설정
    with open("gpt.key", "r") as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key
    openai.seed = 1115

    prompt = "Write a code review given this code : {{input_code}}"
    input_code = patch.get("patch") or patch.get("input_code")
    cur_prompt = prompt.replace("{{input_code}}", input_code)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": cur_prompt}],
        temperature=0.0,
    )

    # ChatGPT의 응답 내용 추출
    answer = response["choices"][0]["message"]["content"].strip()

    return answer


async def evaluate_patch(model, patch, TOP):
    code_review = await asyncio.to_thread(get_codereview, model, patch)
    vanila_gpt_codereview = await asyncio.to_thread(
        get_vanila_gpt_codereview, model, patch
    )

    patch["code_review"] = code_review
    patch["vanila_gpt_codereview"] = vanila_gpt_codereview

    return patch


def save_metrics(patches, input_file_name, model, current_time):
    results = []
    for patch in patches:
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

    # input_file_name = "msg-test-1000.jsonl_gpt-3.5-turbo_20250311_022242.jsonl"
    input_file_name = "msg-test-1000.jsonl_gpt-4o-mini_20250311_020535.jsonl"

    input_file_name = "random_sample_100.jsonl_gpt-4o-mini_20250401_002916.jsonl"
    # input_file_name = "random_sample_100.jsonl_3.jsonl" #for test
    input_file_name = "df_clnl_4.jsonl_gpt-4o-mini_20250401_030404.jsonl"

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
