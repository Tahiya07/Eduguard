#!/usr/bin/env python
"""Deterministic Qwen Bloom-rewrite inference for v4.

Stops on Qwen chat terminators and strips assistant/template residue. This is
an experiment evaluator; it does not modify the production model.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

END_MARKERS=("<|im_end|>","<|endoftext|>")

def prompt(question,target,tokenizer):
    messages=[
      {"role":"system","content":"You are an expert academic assessment editor. Rewrite the complete student-facing academic exam question so that its required cognitive operation matches the requested Bloom level. Preserve the topic, technical entities, quantities, constraints, and academic intent. Do not answer it, explain it, mention Bloom, or use a generic template. Output exactly one exam question."},
      {"role":"user","content":f"Original question:\n{question}\n\nTarget Bloom level:\n{target}\n\nReturn only the rewritten exam question."}
    ]
    if getattr(tokenizer,"chat_template",None):
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"

def clean(x):
    for m in END_MARKERS:
        x=x.split(m,1)[0]
    x=re.sub(r"^\s*(?:answer|response|rewrite|rewritten question|question)\s*:\s*","",x,flags=re.I)
    x=re.sub(r"\s+"," ",x).strip(" \t\"'")
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--question",required=True); ap.add_argument("--target",required=True); ap.add_argument("--max-new-tokens",type=int,default=128); ap.add_argument("--max-input",type=int,default=512); args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"; dtype=torch.float16 if device=="cuda" else torch.float32
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=dtype).to(device).eval()
    p=prompt(args.question,args.target,tok); x=tok(p,return_tensors="pt",truncation=True,max_length=args.max_input); x={k:v.to(device) for k,v in x.items()}
    with torch.no_grad():
        y=model.generate(**x,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    text=clean(tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=False))
    print(json.dumps({"question":args.question,"target":args.target,"rewrite":text},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
