#!/usr/bin/env python3
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    base_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(base_dir, "ckpts", "bert_emotion_bertcls")
    if not os.path.isdir(ckpt_dir):
        print("Checkpoint dir not found:", ckpt_dir)
        print("Please train the model from the notebook first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device).eval()

    id2label = {0: "neg", 1: "pos"}
    print("Loaded checkpoint:", ckpt_dir)
    print("Type a sentence to classify sentiment (q to quit):")

    with torch.no_grad():
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if text.lower() == "q":
                break
            if not text:
                continue
            enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())
            print(f"Pred: {id2label[pred]} | neg={probs[0].item():.4f}, pos={probs[1].item():.4f}")


if __name__ == "__main__":
    main()

# > This movie has a lot of beautiful views, but I do not like the story at all.
# Pred: neg | neg=0.9723, pos=0.0277
# > I really loved this movie, the story was touching and the acting was brilliant!
# Pred: pos | neg=0.0050, pos=0.9950
