#!/usr/bin/env python
"""Generate Bloom rewrite supervision v4 with a stronger teacher and strict QC.

Important:
- Uses only source questions + target Bloom level as teacher inputs.
- Does not use source Bloom level in the generation prompt.
- Reuses the existing v3 TRAIN/VALIDATION source split only to preserve the
  experiment's leakage grouping. Existing v3 targets are never used as labels.
- Does not modify the locked multitask test set.
- Supports either a remote Hugging Face teacher or a local 4-bit Qwen3-14B teacher.
- Local mode is intended for GPU notebooks such as Google Colab and avoids inference-provider credits.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR=Path(__file__).resolve().parent
EXP=SCRIPT_DIR.parent
ROOT=EXP.parents[1]
if str(EXP) not in sys.path: sys.path.insert(0,str(EXP))
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bloom_target_policy_v4 import POLICY_VERSION, build_teacher_messages, validate_candidate, clean_output, canonical_level


def read_jsonl(p):
    rows=[]
    with open(p,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render(messages,tokenizer=None):
    if tokenizer is not None and getattr(tokenizer,"chat_template",None):
        kwargs=dict(tokenize=False,add_generation_prompt=True)
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages,**kwargs)
    parts=[]
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\\n{m['content']}<|im_end|>")
    return "\\n".join(parts)+"\\n<|im_start|>assistant\\n"


def load_hf_teacher(model_id,provider,token_env):
    import os
    from huggingface_hub import InferenceClient

    token=os.getenv(token_env)
    if not token:
        raise RuntimeError(
            f"{token_env} is not set. Create a Hugging Face access token and set "
            f"the environment variable before running teacher generation."
        )

    client=InferenceClient(
        provider=provider,
        token=token,
        timeout=120,
    )
    return client


def load_local_teacher(model_id,device):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

    if device!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("Local Qwen3-14B mode requires a CUDA GPU. Use Colab with a GPU runtime.")

    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token

    quant=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
    )
    model.eval()
    return tok,model


def make_hf_generator(client,model_id,max_new=128,temperature=.7,top_p=.8):
    def generate(source,target,retry=False,repair_reasons=None):
        messages=build_teacher_messages(source,target,retry=retry,repair_reasons=repair_reasons)
        # Qwen3 supports the /no_think soft switch in API prompts.
        messages[-1]["content"] += "\\n\\n/no_think"

        response=client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>","<|endoftext|>"],
        )
        content=getattr(response.choices[0].message,"content",None)
        if not content:
            raise RuntimeError("Hugging Face teacher returned an empty response.")
        return clean_output(str(content))

    return generate


def make_local_generator(tok,model,device,max_input=512,max_new=128,temperature=.7,top_p=.8):
    import torch
    def generate(source,target,retry=False,repair_reasons=None):
        messages=build_teacher_messages(source,target,retry=retry,repair_reasons=repair_reasons)
        try:
            prompt=tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt=tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt += "\n/no_think"
        x=tok(prompt,return_tensors="pt",truncation=True,max_length=max_input)
        x={k:v.to(model.device) for k,v in x.items()}
        with torch.no_grad():
            y=model.generate(
                **x,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        return clean_output(tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=True))
    return generate


def load_llama_teacher(model_path,n_ctx=2048,n_gpu_layers=-1,n_batch=1024,n_threads=None):
    from llama_cpp import Llama
    import os
    if n_threads is None:
        n_threads=max(1,(os.cpu_count() or 4)//2)
    print(
        f"LOADING QWEN3-14B: n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} "
        f"n_batch={n_batch} n_threads={n_threads}",
        flush=True,
    )
    llm=Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )
    print("QWEN3-14B READY", flush=True)
    return llm


def make_llama_generator(llm,max_new=96,temperature=.3,top_p=.8):
    def generate(source,target,retry=False,repair_reasons=None):
        messages=build_teacher_messages(source,target,retry=retry,repair_reasons=repair_reasons)
        messages[-1]["content"] += "\n\n/no_think"
        response=llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>","<|endoftext|>"],
        )
        content=response["choices"][0]["message"].get("content","")
        if not content:
            raise RuntimeError("llama.cpp teacher returned an empty response.")
        return clean_output(str(content))
    return generate


def _parse_json_object(text):
    text=clean_output(str(text or ""))
    fenced=re.search(r"\{.*\}",text,flags=re.S)
    if fenced:
        text=fenced.group(0)
    try:
        obj=json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj,dict) else None


def build_teacher_judge_messages(source_question,target_level,candidate):
    target=canonical_level(target_level) or target_level
    guidance={
        "Remember":"The student should recall or identify information.",
        "Understand":"The student should explain, describe, interpret, summarize, or classify.",
        "Apply":"The student should use a known method, rule, or procedure on a concrete task.",
        "Analyze":"The student should examine parts, relationships, interactions, causes, structure, patterns, or effects.",
        "Evaluate":"The student should make a justified judgment using relevant criteria or evidence.",
        "Create":"The student should design, construct, formulate, develop, or propose a new solution or artifact.",
    }[target]
    system=(
        "You are the final quality-control reviewer for a university Bloom-level rewrite dataset. "
        "Judge one candidate rewrite against its source question and requested target cognitive operation. "
        "Be conservative: reject any candidate that changes the subject matter, invents supplied artifacts, "
        "adds substantive requirements, or changes the intended task beyond what is necessary to express the target level. "
        "New wording needed only to express the cognitive operation is allowed. "
        "A good candidate preserves technical entities, quantities, constraints, and academic intent. "
        "Return JSON only with keys: pass, content_fidelity, scope_fidelity, target_alignment, artifact_fidelity, reason. "
        "Scores are integers from 0 to 2. pass is true only when content_fidelity=2, scope_fidelity=2, "
        "target_alignment=2, and artifact_fidelity=2."
    )
    user=(
        f"Source question:\n{source_question.strip()}\n\n"
        f"Requested target Bloom level:\n{target}\n"
        f"Target cognitive operation:\n{guidance}\n\n"
        f"Candidate rewrite:\n{candidate.strip()}\n\n"
        "Assess the candidate. Do not rewrite it. "
        "Return ONLY the JSON object. Do not think aloud.\\n/no_think"
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]


def make_llama_judge(llm):
    def judge(source,target,candidate):
        try:
            response=llm.create_chat_completion(
                messages=build_teacher_judge_messages(source,target,candidate),
                max_tokens=160,
                temperature=0.0,
                top_p=1.0,
                response_format={
                    "type":"json_object",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "pass":{"type":"boolean"},
                            "content_fidelity":{"type":"integer","enum":[0,1,2]},
                            "scope_fidelity":{"type":"integer","enum":[0,1,2]},
                            "target_alignment":{"type":"integer","enum":[0,1,2]},
                            "artifact_fidelity":{"type":"integer","enum":[0,1,2]},
                            "reason":{"type":"string"},
                        },
                        "required":[
                            "pass","content_fidelity","scope_fidelity",
                            "target_alignment","artifact_fidelity","reason"
                        ],
                    },
                },
                stop=["<|im_end|>","<|endoftext|>"],
            )
            raw=response["choices"][0]["message"].get("content","")
            obj=_parse_json_object(raw)
            if obj is None:
                return {"pass":False,"reason":"judge_parse_error","raw":clean_output(raw)}
            raw_pass=obj.get("pass")
            pass_flag = raw_pass is True or (
                isinstance(raw_pass,str) and raw_pass.strip().lower() in ("true","yes","1")
            ) or (isinstance(raw_pass,(int,float)) and raw_pass==1)
            try:
                scores_ok=all(int(obj.get(k,0))==2 for k in (
                    "content_fidelity","scope_fidelity","target_alignment","artifact_fidelity"
                ))
            except (TypeError,ValueError):
                scores_ok=False
            obj["pass"]=bool(pass_flag and scores_ok)
            return obj
        except Exception as exc:
            return {"pass":False,"reason":"judge_runtime_error:"+str(exc)}
    return judge


def load_semantic():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model=SentenceTransformer("BAAI/bge-small-en-v1.5")
        def sim(a,b):
            e=model.encode([a,b],normalize_embeddings=True)
            return float(np.dot(e[0],e[1]))
        return sim,"BAAI/bge-small-en-v1.5"
    except Exception as e:
        return None,f"unavailable: {e}"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-model",default="Qwen/Qwen3-14B")
    ap.add_argument("--teacher-mode",choices=["hf","local","llama_cpp"],default="hf")
    ap.add_argument("--teacher-provider",default="nscale",help="Hugging Face Inference Provider; use auto for automatic routing")
    ap.add_argument("--teacher-model-path",default="",help="Local GGUF path used only with --teacher-mode llama_cpp")
    ap.add_argument("--n-ctx",type=int,default=2048)
    ap.add_argument("--n-gpu-layers",type=int,default=-1)
    ap.add_argument("--n-batch",type=int,default=1024)
    ap.add_argument("--n-threads",type=int,default=0)
    ap.add_argument("--hf-token-env",default="HF_TOKEN")
    ap.add_argument("--input-v3",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v3")
    ap.add_argument("--output-dir",default="data/bloom_rewrite_versions/bloom_rewrite_synth_v4_1")
    ap.add_argument("--split",choices=["train","validation"],default="train")
    ap.add_argument("--attempts",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--no-semantic",action="store_true")
    ap.add_argument("--min-semantic",type=float,default=.68)
    ap.add_argument("--temperature",type=float,default=.3)
    ap.add_argument("--top-p",type=float,default=.8)
    ap.add_argument("--device",default="cuda",choices=["cuda","cpu"])
    ap.add_argument("--checkpoint-every",type=int,default=25)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()

    random.seed(args.seed)
    inp=Path(args.input_v3)
    out=Path(args.output_dir)
    source_rows=read_jsonl(inp/f"{args.split}.jsonl")
    pairs={}
    for r in source_rows:
        q=r.get("source_question")
        t=canonical_level(r.get("target_bloom_level"))
        if q and t:
            pairs[(str(r.get("source_id") or sha(q)[:16]),q,t)]=r
    items=list(pairs.values())
    random.shuffle(items)
    if args.limit:
        items=items[:args.limit]

    if args.teacher_mode=="llama_cpp":
        if not args.teacher_model_path:
            raise SystemExit("--teacher-model-path is required with --teacher-mode llama_cpp")
        llm=load_llama_teacher(args.teacher_model_path,args.n_ctx,args.n_gpu_layers,args.n_batch,args.n_threads or None)
        generate=make_llama_generator(llm,max_new=96,temperature=args.temperature,top_p=args.top_p)
        judge=make_llama_judge(llm)
    elif args.teacher_mode=="local":
        tok,model=load_local_teacher(args.teacher_model,args.device)
        generate=make_local_generator(tok,model,args.device,max_new=128,temperature=args.temperature,top_p=args.top_p)
        judge=None
    else:
        client=load_hf_teacher(args.teacher_model,args.teacher_provider,args.hf_token_env)
        generate=make_hf_generator(client,args.teacher_model,max_new=128,temperature=args.temperature,top_p=args.top_p)
        judge=None
    sim_fn,sim_name=(None,"disabled") if args.no_semantic else load_semantic()

    accepted=[]
    failures=Counter()
    attempts_used=[]
    judge_passes=0
    judge_rejections=0
    existing_path=out/f"{args.split}.jsonl"
    completed_keys=set()
    if args.resume and existing_path.exists():
        report_path=out/f"{args.split}_report.json"
        if report_path.exists():
            previous=json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("policy_version") != POLICY_VERSION or previous.get("dataset_version") != "bloom_rewrite_synth_v4_1":
                raise SystemExit(
                    "Refusing to resume an incompatible dataset: the existing output was generated with "
                    f"policy={previous.get('policy_version')} dataset={previous.get('dataset_version')}. "
                    "Use a fresh output directory or delete the old v4 output."
                )
        existing=read_jsonl(existing_path)
        accepted.extend(existing)
        completed_keys={(str(r.get("source_id")),canonical_level(r.get("target_bloom_level"))) for r in existing}
        print(f"RESUME: loaded {len(existing)} accepted rows from {existing_path}")

    total_items=len(items)
    run_started=time.perf_counter()

    for i,row in enumerate(items,1):
        source=row["source_question"]
        target=row["target_bloom_level"]
        item_started=time.perf_counter()
        print(
            f"[{i}/{total_items}] START target={target} "
            f"source={source[:100].replace(chr(10),' ')}",
            flush=True,
        )
        source_key=(str(row.get("source_id") or sha(source)[:16]),canonical_level(target))
        if source_key in completed_keys:
            continue
        best=None
        repair_reasons=None

        for attempt in range(args.attempts):
            print(f"[{i}/{total_items}] attempt {attempt+1}/{args.attempts} generating", flush=True)
            gen_started=time.perf_counter()
            try:
                candidate=generate(
                    source,
                    target,
                    retry=attempt>0,
                    repair_reasons=repair_reasons,
                )
            except Exception as exc:
                failures["TEACHER_API_ERROR"]+=1
                print(f"teacher_error example={i} attempt={attempt+1}: {exc}")
                continue
            gen_seconds=time.perf_counter()-gen_started
            print(f"[{i}/{total_items}] attempt {attempt+1} generated in {gen_seconds:.1f}s", flush=True)
            semantic=sim_fn(source,candidate) if sim_fn and candidate else None
            v=validate_candidate(
                source,
                target,
                candidate,
                semantic_similarity=semantic,
                min_semantic_similarity=args.min_semantic,
            )
            judge_result=None

            # In llama.cpp mode, let the 14B teacher adjudicate candidates that
            # have only soft lexical/structure warnings. Hard safety/content
            # violations are rejected before adjudication.
            hard_prefixes=(
                "EMPTY_OUTPUT","META_OR_ANSWER_LANGUAGE","INVALID_EXAM_QUESTION_FORM",
                "INVENTED_ARTIFACT_REFERENCE","SCOPE_EXPANSION","SCOPE_CONTENT_ADDITION",
                "PROTECTED_SPAN_LOSS","LOW_SEMANTIC_SIMILARITY","CLASSIFIER_TARGET_MISMATCH",
                "CLASSIFIER_LOW_CONFIDENCE","MULTIPLE_QUESTIONS","TOO_LONG",
                "NEAR_SOURCE_COPY","EXACT_SOURCE_COPY"
            )
            hard_reasons=[x for x in v.reasons if x.upper().startswith(hard_prefixes)]

            if judge is not None and not hard_reasons:
                print(f"[{i}/{total_items}] judge running", flush=True)
                judge_started=time.perf_counter()
                judge_result=judge(source,target,candidate)
                judge_seconds=time.perf_counter()-judge_started
                print(
                    f"[{i}/{total_items}] judge finished in {judge_seconds:.1f}s "
                    f"pass={judge_result.get('pass',False)} reason={judge_result.get('reason','')}",
                    flush=True,
                )
                if judge_result.get("pass",False):
                    judge_passes+=1
                    # Teacher adjudication resolves soft warnings such as
                    # lexical overlap or rigid keyword-based target cues.
                    v.ok=True
                    v.failure_category=""
                else:
                    judge_rejections+=1
                    v.reasons.append("teacher_judge:" + str(judge_result.get("reason","rejected")))
                    v.failure_category="TEACHER_JUDGE_REJECTION"
                    v.ok=False

            if v.ok:
                best=(candidate,v,attempt+1,judge_result)
                print(
                    f"[{i}/{total_items}] ACCEPTED on attempt {attempt+1} "
                    f"item_time={time.perf_counter()-item_started:.1f}s",
                    flush=True,
                )
                break

            failures[v.failure_category or "QUALITY_REJECTION"]+=1
            repair_reasons=v.reasons
            print(
                f"[{i}/{total_items}] REJECTED attempt {attempt+1}: "
                f"{v.failure_category or 'QUALITY_REJECTION'} "
                f"reasons={'; '.join(v.reasons[:3])}",
                flush=True,
            )

        if best is None:
            attempts_used.append(args.attempts)
            print(
                f"[{i}/{total_items}] REJECTED after {args.attempts} attempts "
                f"item_time={time.perf_counter()-item_started:.1f}s",
                flush=True,
            )
            continue

        candidate,v,ntry,judge_result=best
        attempts_used.append(ntry)
        src_id=str(row.get("source_id") or sha(source)[:16])

        rec={
            "example_id":sha(f"v4.1|{src_id}|{target}|{candidate}")[:16],
            "source_id":src_id,
            "group_id":row.get("group_id"),
            "split":args.split,
            "source_question":source,
            "source_bloom_level":row.get("source_bloom_level"),
            "target_bloom_level":target,
            "target_rewrite":candidate,
            "transformation_type":f"{row.get('source_bloom_level')}->{target}",
            "synthetic_or_original":"synthetic_teacher_generated",
            "synthetic":True,
            "dataset_version":"bloom_rewrite_synth_v4_1",
            "policy_version":POLICY_VERSION,
            "quality_status":"pass",
            "validation":v.__dict__,
            "teacher_judge":judge_result,
            "teacher_model":args.teacher_model,
            "teacher_provider":args.teacher_provider if args.teacher_mode=="hf" else None,
            "generator_inputs":["source_question","target_bloom_level"],
            "teacher_attempts":ntry,
        }
        accepted.append(rec)

        if args.checkpoint_every and i%args.checkpoint_every==0:
            write_jsonl(existing_path,accepted)
            print(f"CHECKPOINT {i}/{len(items)} accepted={len(accepted)} elapsed={time.perf_counter()-run_started:.1f}s", flush=True)

    write_jsonl(out/f"{args.split}.jsonl",accepted)

    report={
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_version":"bloom_rewrite_synth_v4_1",
        "policy_version":POLICY_VERSION,
        "teacher_model":args.teacher_model,
        "teacher_mode":args.teacher_mode,
        "teacher_provider":args.teacher_provider if args.teacher_mode=="hf" else None,
        "hf_token_env":args.hf_token_env,
        "source_dataset":str(inp),
        "split":args.split,
        "source_pairs":len(items),
        "accepted":len(accepted),
        "rejected":len(items)-len(accepted),
        "acceptance_rate":len(accepted)/len(items) if items else 0,
        "failure_counts":dict(failures),
        "semantic_model":sim_name,
        "min_semantic":args.min_semantic,
        "attempts":args.attempts,
        "max_new_tokens":96,
        "judge_max_tokens":160,
        "teacher_judge_enabled":args.teacher_mode=="llama_cpp",
        "teacher_judge_passes":judge_passes,
        "teacher_judge_rejections":judge_rejections,
        "elapsed_seconds":round(time.perf_counter()-run_started,2),
        "transformation_counts":dict(Counter(x["transformation_type"] for x in accepted)),
    }
    (out/f"{args.split}_report.json").write_text(
        json.dumps(report,indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
