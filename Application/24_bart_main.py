# %%
import os
import torch
import datasets
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.models.bart.modeling_bart import shift_tokens_right

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "ckpts", "bart_cnn_summary")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs_cnn")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 1. 强制环境设置：解决 CUDA/NCCL 错误 ---

# 1.1 禁用 NCCL P2P/IB
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# 1.2 隔离 GPU 3，确保 Trainer 只在单个 GPU 上运行 (物理 GPU 3 -> 逻辑 cuda:0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["NCCL_DEBUG"] = "INFO" 
torch.cuda.empty_cache()

# --- 2. 模型和分词器初始化 ---

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device set: {device}")

MODEL_NAME = 'facebook/bart-large-cnn'
print(f"Loading model: {MODEL_NAME}")

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# --- 3. 数据准备：切换到 CNN/Daily Mail 数据集 ---

print("Loading and preparing CNN/Daily Mail dataset...")
raw_datasets = datasets.load_dataset("cnn_dailymail", "3.0.0") 

# 截取一个子集，加快演示速度
_train = raw_datasets['train'].select(range(2000))
_val = raw_datasets['validation'].select(range(500))

dataset = datasets.DatasetDict({
    "train": _train,
    "validation": _val,
})

def convert_to_features(example_batch):
    # 1. Encoder Input (Source Text: article)
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['article'],
        padding='max_length',
        max_length=1024,
        truncation=True
    )
    
    # 2. Target Summary Input (Target Text: highlights)
    target_encodings = tokenizer.batch_encode_plus(
        example_batch['highlights'],
        padding='max_length',
        max_length=128,
        truncation=True
    )
    
    labels = torch.tensor(target_encodings['input_ids'])
    
    # 3. 构造 Decoder Input
    decoder_input_ids = shift_tokens_right(
        labels, 
        model.config.pad_token_id, 
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # 4. 构造 Labels
    labels[labels == model.config.pad_token_id] = -100
    
    encodings = {
        'input_ids': torch.tensor(input_encodings['input_ids']),
        'attention_mask': torch.tensor(input_encodings['attention_mask']),
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }
    return encodings

# 应用特征转换，并移除原始列
dataset = dataset.map(convert_to_features, batched=True, remove_columns=['article', 'highlights', 'id'])
# 设置格式为 PyTorch Tensor
columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask']
dataset.set_format(type='torch', columns=columns)

print("Dataset prepared successfully.")

# --- 4. 训练参数配置和 Trainer 实例化 (关键优化在此) ---

training_args = Seq2SeqTrainingArguments(
    output_dir=CKPT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    per_device_eval_batch_size=1,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=10,
    eval_strategy="steps", 
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), 
    
    # ---------------- 关键修改 ----------------
    # 1. 只保留一个最好的检查点
    save_total_limit=1, 
    # 2. 训练结束后加载性能最好的模型
    load_best_model_at_end=True, 
    # 3. 定义评判“最好”模型的指标，摘要任务常用 ROUGE 或 损失 (Loss)
    # 这里选择 Loss (越小越好)
    metric_for_best_model="eval_loss",
    # 4. 确保最佳模型是基于最小损失（默认是越小越好，但明确设置更安全）
    greater_is_better=False 
    # ------------------------------------------
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer, 
)

# --- 5. 开始训练 ---
print("Starting training with CNN/Daily Mail dataset...")
# 训练完成后，最好的检查点将作为最终模型保存在 output_dir 中
trainer.train()

print("Training finished.")
print(f"Only the best model checkpoint (based on {training_args.metric_for_best_model}) is saved in {training_args.output_dir}")