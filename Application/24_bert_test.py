# %%
import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate # 用于计算 ROUGE 分数
import numpy as np

# --- 1. 路径和环境配置 (基于您的最终确认) ---

# 1.1 路径设置 (假设测试脚本与训练脚本在同一目录)
try:
    # 尝试获取当前文件路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果在交互式环境 (如 Jupyter/IPython) 运行
    SCRIPT_DIR = os.getcwd() 
    print(f"Warning: Running in interactive mode. Assuming script directory is: {SCRIPT_DIR}")

# 模型加载路径
CKPT_DIR = os.path.join(SCRIPT_DIR, "ckpts", "bart_cnn_summary")
MODEL_NAME = 'facebook/bart-large-cnn' 

# 1.2 强制环境设置：解决 CUDA/NCCL 错误
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# 隔离 GPU 3，确保 Trainer 只在单个 GPU 上运行 (物理 GPU 3 -> 逻辑 cuda:0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["NCCL_DEBUG"] = "INFO" 
torch.cuda.empty_cache()

# --- 2. 模型和分词器初始化 ---

# 由于设置了 CUDA_VISIBLE_DEVICES="3"，目标设备应为 cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device set: {device}")
print(f"Loading model from: {CKPT_DIR} onto device: {device}")

try:
    # 加载训练好的模型
    model = BartForConditionalGeneration.from_pretrained(CKPT_DIR).to(device)
    # 加载对应的分词器
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"\n 错误：无法加载模型或分词器。请检查路径 {CKPT_DIR} 是否存在保存的模型文件。")
    print(f"详细错误: {e}")
    # 确保加载失败时，脚本不会继续尝试调用模型
    model = None 
    tokenizer = None
    exit()

# --- 3. 摘要生成函数 (已修复 length_penalty 错误) ---

def generate_summary(text):
    """使用加载的模型和分词器生成摘要。"""
    
    inputs = tokenizer(
        [text], 
        max_length=1024, 
        return_tensors='pt', 
        truncation=True
    ).to(device)
    
    # 修复了 length_penalty=None 的错误，设置为 0.6
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=130,      
        min_length=30,       
        num_beams=4,         
        do_sample=False,     
        early_stopping=True,
        length_penalty=0.6 # <--- 关键修复点
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- 4. 自动化测试和对比分析 ---

# 定义测试文章和参考摘要
TEST_ARTICLE = """ A new study published in the journal Nature Astronomy suggests that there may be a vast, hidden ocean beneath the icy crust of Pluto's largest moon, Charon. Researchers from the Southwest Research Institute analyzed data collected by NASA's New Horizons mission during its 2015 flyby. They found deep fissures and surface fractures that indicate Charon's surface expanded at some point in its history. This expansion is consistent with a subsurface ocean that froze, causing the moon's outer layers to stretch and crack. If water were present, it would offer another potential location for astrobiological interest in the Kuiper Belt, far beyond Neptune. The current model suggests the ocean may be up to 100 kilometers thick and is likely frozen solid now, but liquid water may have persisted for billions of years due to radioactive decay heating the interior."""

REFERENCE_SUMMARY = """Scientists analyzed New Horizons data from Charon and found deep fissures and fractures, suggesting the moon's surface expanded due to a vast, subsurface ocean. The ocean may have been liquid for billions of years, heated by radioactive decay, and is now likely frozen solid, providing another potential location for astrobiological study."""

print("\n==================================================")
print("             🚀 自动化测试和 ROUGE 分析 🚀")
print("==================================================")

# 1. 生成模型摘要
try:
    model_summary = generate_summary(TEST_ARTICLE)
except Exception as e:
    print(f"\n❌ 致命错误：模型生成摘要失败。详细错误: {e}")
    model_summary = "摘要生成失败"


print("\n--- 原始文章 ---")
print(TEST_ARTICLE[:200] + "...")

print("\n--- 标准参考摘要 ---")
print(REFERENCE_SUMMARY)

print("\n--- 您的模型摘要 ---")
print(model_summary)


# 2. 计算 ROUGE 分数 (仅当摘要成功生成时)
if model_summary != "摘要生成失败":
    try:
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=[model_summary], references=[REFERENCE_SUMMARY])

        print("\n--- ROUGE 对比分析结果 (F1 Score) ---")

        # 格式化输出 ROUGE 结果
        for key, value in results.items():
            if isinstance(value, dict) and 'fmeasure' in value:
                f1_score = value['fmeasure'] * 100 # 转换为百分比
            elif isinstance(value, float):
                f1_score = value * 100
            else:
                continue
                
            print(f"| {key.ljust(9)} | {f1_score:.2f}%")

        print("--------------------------------------------------")
        print("ROUGE F1 Score 越高，表明您的摘要与参考摘要越相似。")
    except Exception as e:
        print(f"\nROUGE 评估计算失败：请确保您已安装 'evaluate' 和 'rouge-score'。详细错误: {e}")
    
print("==================================================")

# --- 5. 交互式循环 (接续) ---

print("\n--- 交互式摘要生成模式 ---")
print("您可以继续输入自定义文章进行测试，或输入 'exit' 退出程序。\n")

while True:
    try:
        article_text = input("文章内容 >>> ")
        
        if article_text.lower() in ['exit', 'quit']:
            print("退出摘要程序。")
            break
        
        if not article_text.strip():
            print("请输入有效的文章内容。\n")
            continue

        print("\n⏳ 正在生成摘要...")
        summary_result = generate_summary(article_text)
        
        print("\n--- 摘要结果 ---")
        print(summary_result)
        print("------------------\n")
        
    except KeyboardInterrupt:
        print("\n捕获到中断信号，退出程序。")
        break
    except Exception as e:
        print(f"\n处理错误: {e}")
        print("请重试或检查输入。\n")