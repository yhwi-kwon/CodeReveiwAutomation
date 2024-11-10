import openai
import json
import re
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_review_feedback(model, patch):
    # OpenAI API 키 설정
    with open('gpt.key', 'r') as key_file:
        api_key = key_file.readline().strip()
    openai.api_key = api_key

    diff_text = patch
    prompt = open('prompt/diff_estimation_prompt.txt').read()
    cur_prompt = prompt.replace('{{diff_text}}', diff_text)

    # ChatGPT API에 질문하여 코드리뷰 필요 여부 판단
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": cur_prompt}
        ]
    )

    # ChatGPT의 응답 내용 추출
    answer = response['choices'][0]['message']['content'].strip()

    # Regular expression to extract the number
    match = re.search(r'Code Review Required: (\d)', answer)

    # Extract and print the result if found
    score = 3
    if match:
        score = int(match.group(1))
    else:
        print("No score found")
        print(answer)

    return (1 if score >= 3 else 0), score

def main():
    # 모델 설정
    model = "gpt-3.5-turbo"
    #model = "gpt-4o-mini"
    #model = "gpt-4o"
    #model = "gpt-4-turbo"
    input_file_name = 'diff_estimation_1_100.jsonl'
    
    with open(f'data/{input_file_name}', 'r') as file:
        patches = [json.loads(line) for line in file]

    # 현재 시간 추가
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # output 파일 생성
    with open(f'output/{input_file_name}_{model}_{current_time}.jsonl', 'w') as output_file:
        try:
            y_true = []
            y_pred = []

            for patch in tqdm(patches, desc="Processing patches"):
                y_true.append(int(patch['y']))
                review_needed, score = get_review_feedback(model, patch['patch'])
                y_pred.append(int(review_needed))
                
                # 기존 데이터에 y_pred 추가
                patch['y_pred'] = review_needed
                patch['y_pred_score'] = score
                
                # JSON 형식으로 저장
                output_file.write(json.dumps(patch) + '\n')

            # Precision, Recall, F1, Accuracy 계산
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print(f"Accuracy: {accuracy:.2f}")

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
