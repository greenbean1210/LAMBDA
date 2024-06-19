import PyPDF2
from transformers import T5ForConditionalGeneration, T5Tokenizer
import tkinter as tk
from tkinter import filedialog
import re

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# 텍스트 전처리 함수
def preprocess_text(text):
    # 불필요한 공백과 특수문자 제거
    text = re.sub(r'\s+', ' ', text)  # 여러 개의 공백을 하나로
    text = re.sub(r'[^A-Za-z0-9\s.,?!]', '', text)  # 알파벳, 숫자, 공백, ., ,, ?, ! 외의 문자 제거
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)  # 문장 부호 앞의 공백 제거
    text = text.strip()  # 양 끝의 공백 제거
    return text

# 문장 단위로 끊기
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# 질문 생성 파이프라인 설정
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")

def generate_questions(context):
    input_text = "generate questions: " + context + " </s>"
    print(f"Input to model: {input_text[:500]}")  # 디버깅을 위해 추가
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated questions: {questions}")  # 디버깅을 위해 추가
    return questions.split("<sep>")

# 질문 변형 함수 작성
def modify_question(question):
    if "What" in question:
        question = question.replace("What", "Which")
    return question

def create_variant_questions(original_questions):
    variant_questions = []
    for question in original_questions:
        new_question = modify_question(question)
        variant_questions.append(new_question)
    return variant_questions

# 사용자 인터페이스 구축
def open_file():
    file_path = filedialog.askopenfilename()
    text = extract_text_from_pdf(file_path)
    processed_text = preprocess_text(text)
    print(f"Extracted text: {processed_text[:500]}")  # 첫 500자를 출력하여 확인
    sentences = split_into_sentences(processed_text)
    
    original_questions = []
    for sentence in sentences:
        if len(sentence) > 20:  # 너무 짧은 문장은 생략
            questions = generate_questions(sentence)
            original_questions.extend(questions)
    
    variant_questions = create_variant_questions(original_questions)
    display_questions(original_questions, variant_questions)

def display_questions(original_questions, variant_questions):
    print("Original Questions:")
    for question in original_questions:
        print(f"- {question.strip()}")

    print("\nVariant Questions:")
    for question in variant_questions:
        print(f"- {question.strip()}")

root = tk.Tk()
root.title("영어 모의고사 변형 문제 생성기")
open_button = tk.Button(root, text="파일 열기", command=open_file)
open_button.pack()
root.mainloop()