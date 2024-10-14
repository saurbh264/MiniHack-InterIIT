from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import requests
import json
from dotenv import load_dotenv
import os


# MoonDream Image2Text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

model = model.to(device)


# Qwen Text2Text
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model1 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer1 = AutoTokenizer.from_pretrained(model_name)


def model_func(image, question, n):
    
    load_dotenv()

    # MoonDream inference
    enc_image = model.encode_image(image).to(device)
    context = model.answer_question(enc_image, "Give the product features in concise manner", tokenizer)

    # Qwen inference
    prompt = f"""{context} {question}
    What does it want me to show, answer in short and concise manner ?
    """
    # prompt = f"""{context} {question}
    # What does it want me to show, answer as short as possible?
    # """
    messages = [
        {"role": "system", "content": "You are helpful assisstant that give best search query"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer1.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model1_inputs = tokenizer1([text], return_tensors="pt").to(model1.device)

    generated_ids = model1.generate(
        **model1_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model1_inputs.input_ids, generated_ids)
    ]

    query = tokenizer1.batch_decode(generated_ids, skip_special_tokens=True)[0]


    url = "https://api.serphouse.com/serp/live"
    payload = {
        "data": {
            "q": query, #Change this to relevant search query
            "domain": "in.yahoo.com",
            "loc": "Abernathy,Texas,United States",
            "lang": "lang_en",
            "device": "desktop",
            "serp_type": "image",
            "page": "1",
            "verbatim": "0"
        }
    }
    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        'authorization': f"Bearer {os.getenv('API_KEY')}"  #APi key
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    result = response.text
    result = json.loads(result)

    images = [dic['original'] for dic in result['results']['results']]
    
    return images[:n]

