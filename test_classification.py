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

    # Regular expression to extract the categories, subcategories, and reasons
    primary_category_match = re.search(r'Primary Category: (.+)', answer)
    primary_subcategory_match = re.search(r'Primary Subcategory: (.+)', answer)
    primary_reason_match = re.search(r'Primary Reason: (.+)', answer)

    secondary_category_match = re.search(r'Secondary Category: (.+)', answer)
    secondary_subcategory_match = re.search(
        r'Secondary Subcategory: (.+)', answer)
    secondary_reason_match = re.search(r'Secondary Reason: (.+)', answer)

    tertiary_category_match = re.search(r'Tertiary Category: (.+)', answer)
    tertiary_subcategory_match = re.search(
        r'Tertiary Subcategory: (.+)', answer)
    tertiary_reason_match = re.search(r'Tertiary Reason: (.+)', answer)

    # Extract and print the result if found
    primary_category = "unknown"
    primary_subcategory = "unknown"
    primary_reason = "unknown"
    if primary_category_match:
        primary_category = primary_category_match.group(1).strip().lower()
    if primary_subcategory_match:
        primary_subcategory = primary_subcategory_match.group(1).strip()
    if primary_reason_match:
        primary_reason = primary_reason_match.group(1).strip()

    secondary_category = "unknown"
    secondary_subcategory = "unknown"
    secondary_reason = "unknown"
    if secondary_category_match:
        secondary_category = secondary_category_match.group(1).strip().lower()
    if secondary_subcategory_match:
        secondary_subcategory = secondary_subcategory_match.group(1).strip()
    if secondary_reason_match:
        secondary_reason = secondary_reason_match.group(1).strip()

    tertiary_category = "unknown"
    tertiary_subcategory = "unknown"
    tertiary_reason = "unknown"
    if tertiary_category_match:
        tertiary_category = tertiary_category_match.group(1).strip().lower()
    if tertiary_subcategory_match:
        tertiary_subcategory = tertiary_subcategory_match.group(1).strip()
    if tertiary_reason_match:
        tertiary_reason = tertiary_reason_match.group(1).strip()

    return primary_category, primary_subcategory, primary_reason, secondary_category, secondary_subcategory, secondary_reason, tertiary_category, tertiary_subcategory, tertiary_reason


def evaluate_patch(model, patch, TOP):
    y_true_categories = [category.strip().lower() for category in patch['first_category'].split(
        ';')]  # Split multiple categories and convert to lowercase
    input_code = patch['input_code']
    primary_category, primary_subcategory, primary_reason, secondary_category, secondary_subcategory, secondary_reason, tertiary_category, tertiary_subcategory, tertiary_reason = get_classification(
        model, input_code)

    primary_category = primary_category.strip().lower() or "refactoring"
    secondary_category = secondary_category.strip().lower() or "refactoring"
    tertiary_category = tertiary_category.strip().lower() or "refactoring"

    patch['primary_category'] = primary_category
    patch['primary_subcategory'] = primary_subcategory
    patch['primary_reason'] = primary_reason
    patch['secondary_category'] = secondary_category
    patch['secondary_subcategory'] = secondary_subcategory
    patch['secondary_reason'] = secondary_reason
    patch['tertiary_category'] = tertiary_category
    patch['tertiary_subcategory'] = tertiary_subcategory
    patch['tertiary_reason'] = tertiary_reason

    # Check if primary_category, secondary_category, or tertiary_category is in y_true categories
    y_true = y_true_categories[0]
    y_pred = primary_category

    for category in y_true_categories:
        if TOP >= 1 and primary_category == category:  # Already converted to lowercase for comparison
            y_true = category
            y_pred = primary_category
            break
        elif TOP >= 2 and secondary_category == category:
            y_true = category
            y_pred = secondary_category
            break
        elif TOP >= 3 and tertiary_category == category:
            y_true = category
            y_pred = tertiary_category
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
    print(
        f"Metrics saved to output/1.2/{input_file_name}_{model}_{current_time}.result")


def main():
    # 모델 설정
    # model = "gpt-3.5-turbo"
    model = "gpt-4o-mini"
    # model = "gpt-4o"
    # model = "gpt-4-turbo"
    # model = "o3-mini"
    # model = "o1-mini"

    TOP = 1

    input_file_name = 'df_clnl_4.jsonl'

    with open(f'data/{input_file_name}', 'r') as file:
        patches = [json.loads(line) for line in file]

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(f'output/1.2/{input_file_name}_{model}_{current_time}.jsonl', 'w') as output_file:

        y_true_list = []
        y_pred_list = []

        for patch in tqdm(patches, desc="Processing patches"):
            try:
                y_true, y_pred, updated_patch = evaluate_patch(
                    model, patch, TOP)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

                output_file.write(json.dumps(updated_patch) + '\n')
            except Exception as e:
                print("Error:", e)
        save_metrics(y_true_list, y_pred_list,
                     input_file_name, model, current_time)


if __name__ == "__main__":
    main()
