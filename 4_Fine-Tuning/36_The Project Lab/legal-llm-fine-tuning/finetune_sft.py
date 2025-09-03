import os
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from transformers import BitsAndBytesConfig


# prepatre training data for supervised fine-tuning

def prepare_training_data():
    os.makedirs("data/train", exist_ok=True)

    gold_files = [f"data/gold/{f}" for f in os.listdir("data/gold") if f.endswith(".json")]
    gold_data = []

    for f in gold_files:
        df = pd.read_json(f)
        gold_data.extend([
            {
                "text": f"<s>[INST] {row['text']} [/INST] {row['label']} </s>",
                "task": "clause_extraction" if "clause" in row else "qa"
            } for _, row in df.iterrows()
        ])
    
    chunk_files = [f"data/processes/chunks/{f}" for f in os.listdir("data/processed/chunks") if f.endswith(".txt")]
    chunk_data = []

    for f in chunk_files[:50]:
        with open(f, "r") as file:
            text = file.read()
            chunk_data.append({
                "text": f"<s>[INST] Provide a summary of the text {text} [/INST] Summary: {text[:200]}...</s>",
                "task": "summarization"
            })
    
    all_data = gold_data + chunk_data
    dataset = Dataset.from_list(all_data)
    dataset.save_to_disk("data/train/legal_sft")

    print(f"Prepared {len(all_data)} training samples.")


# fine-tuning logic LLama3.2:1b using LoRA and 4 bit quantization

def fine_tune_model():

    model_name = "meta-llama/Llama-3.2-1B"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="mps"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_from_disk("data/train/legal_sft")

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                    remove_columns=["text", "task"])
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="output/finetuned_legal_model",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        report_to="none",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    trainer.train()

    model.save_pretrained("output/finetuned_legal_llama")
    tokenizer.save_pretrained("output/finetuned_legal_llama")
    print("SFT complete and model saved.")



if __name__ == "__main__":
    prepare_training_data()
    fine_tune_model()
    