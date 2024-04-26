import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

wandb_API = os.getenv('WANDB_API_KEY')

# Login to Weights and Biases
wandb.login(key=wandb_API)

# Initialize Weights and Biases
wandb.init(project="phi3-finetuning")

# Local paths for the model and tokenizer
local_model_dir = "C:\\Users\\rochi\\Desktop\\CV\\Code\\Finetune COBOL\\models\\phi-3-mini-4k-instruct-gguf"
fine_tuned_model_dir = "C:\\Users\\rochi\\Desktop\\CV\\Code\\Finetune COBOL\\models\\phi-3-mini-4k-instruct-gguf-finetuned"

# Function to check if the required files exist in the directory
def check_required_files(directory):
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(directory, file)):
            return False
    return True

# Check if the model is already downloaded, otherwise download it from the Hugging Face Hub
if not check_required_files(local_model_dir):
    print("Downloading model files from the Hugging Face Hub...")
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True).save_pretrained(local_model_dir)
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True).save_pretrained(local_model_dir)
    print("Model files downloaded and stored locally.")
else:
    print("Model files already exist locally. Skipping download.")

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=20,
    target_modules="all-linear",
    lora_dropout=0.02,
    bias="none",
    task_type="CAUSAL_LM"
)
model = AutoModelForCausalLM.from_pretrained(local_model_dir, quantization_config=bnb_config, trust_remote_code=True)
model = get_peft_model(model, lora_config)
if torch.cuda.is_available():
    model.cuda()  # Move model to GPU if available
model.eval()
print("Model loaded on device:", model.device)


# SQLDataset class for handling the Text2SQL data
class SQLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        output = record["output"]
        query = record["query"]
        schema = record["schema"]
        input = f"Schema: {schema}\nInstructions: {query}\nAnswer: {output}"
        encoding = self.tokenizer(input, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {'input_ids': encoding.input_ids.squeeze(0), 'attention_mask': encoding.attention_mask.squeeze(0)}

# Load the training and evaluation datasets
train_file_path = "data/SQL/train.jsonl"
eval_file_path = "data/SQL/test.jsonl"
train_dataset = SQLDataset(file_path=train_file_path, tokenizer=tokenizer)
eval_dataset = SQLDataset(file_path=eval_file_path, tokenizer=tokenizer)

# Set up the data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./training",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=10,
    evaluation_strategy="epoch",
    report_to="wandb"  # Report to Weights and Biases
)

# Set up the trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained(fine_tuned_model_dir)

# Function to generate text
def generate_text(prompt, model, max_new_tokens=2048):
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding['input_ids'].to(model.device)  # Ensure tensor is on the same device as model
    attention_mask = encoding['attention_mask'].to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Finish the Weights and Biases run
wandb.finish()