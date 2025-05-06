from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

model_name = "deepseek-ai/deepseek-llm-7b-chat"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")

dataset = load_dataset("json", data_files="opensubs_chatml.jsonl", split="train")

def format_chat(example):
    output = ""
    for msg in example['messages']:
        role = msg["role"]
        content = msg["content"]
        output += f"<|{role}|>\n{content}\n"
    return {"text": output.strip()}

def tokenize_function(examples):
    return tokenizer(
        examples["text"],  # or "messages"
        truncation=True,
        padding="max_length",
        max_length=512,
    )

dataset = dataset.map(format_chat)
tokenized_dataset = dataset.map(tokenize_function, batched=True).remove_columns(['messages','text'])
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    eval_strategy="no",
    save_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    remove_unused_columns=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./Finetunned_Deepseek_Model")
tokenizer.save_pretrained('./Finetunned_Deepseek_Tokeneizer')
