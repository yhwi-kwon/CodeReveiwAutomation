import openai
import json
import re
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


def get_classification(model, input_code):
    # OpenAI API 키 설정
    with open('gpt.key', 'r') as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key

    prompt = open('prompt/classification_prompt.txt').read()
    cur_prompt = prompt.replace('{{input_code}}', input_code)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": cur_prompt}
        ]
    )

    # ChatGPT의 응답 내용 추출
    answer = response['choices'][0]['message']['content'].strip()

    # bold markdown 제거
    answer = answer.replace('**', '')

    # Regular expression to extract the category, subcategory, and reason
    main_category_match = re.search(r'Main Category: (.+)', answer)
    subcategory_match = re.search(r'Subcategory: (.+)', answer)
    reason_match = re.search(r'Reason: (.+)', answer)

    # Extract and print the result if found
    main_category = "unknown"
    subcategory = "unknown"
    reason = "unknown"
    if main_category_match:
        main_category = main_category_match.group(1).strip().lower()
    if subcategory_match:
        subcategory = subcategory_match.group(1).strip()
    if reason_match:
        reason = reason_match.group(1).strip()

    return main_category, subcategory, reason


def evaluate_patch(model, patch):
    y_true_categories = patch['first_category'].split(
        ';')  # Split multiple categories
    input_code = patch['input_code']
    y_pred, subcategory, reason = get_classification(model, input_code)

    patch['y_pred'] = y_pred
    patch['subcategory'] = subcategory
    patch['reason'] = reason

    # Check if y_pred is in y_true categories
    y_true = "unknown"
    for category in y_true_categories:
        if y_pred == category.strip().lower():  # Convert to lowercase for comparison
            y_true = category.strip().lower()
            break

    return y_true, y_pred, patch


def save_metrics(y_true_list, y_pred_list, input_file_name, model, current_time):
    precision = precision_score(y_true_list, y_pred_list, average='macro')
    recall = recall_score(y_true_list, y_pred_list, average='macro')
    f1 = f1_score(y_true_list, y_pred_list, average='macro')
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
        "Accuracy": accuracy
    }

    with open(f'output/1.2/{input_file_name}_{model}_{current_time}.result', 'w') as result_file:
        json.dump(result, result_file, indent=4)


def main():
    # 모델 설정
    model = "gpt-3.5-turbo"

    input_file_name = 'df_clnl_4.jsonl'

    with open(f'data/{input_file_name}', 'r') as file:
        patches = [json.loads(line) for line in file]

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(f'output/1.2/{input_file_name}_{model}_{current_time}.jsonl', 'w') as output_file:

        y_true_list = []
        y_pred_list = []

        for patch in tqdm(patches, desc="Processing patches"):
            try:
                y_true, y_pred, updated_patch = evaluate_patch(model, patch)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

                output_file.write(json.dumps(updated_patch) + '\n')
            except Exception as e:
                print("Error:", e)
        save_metrics(y_true_list, y_pred_list,
                     input_file_name, model, current_time)


if __name__ == "__main__":
    main()
