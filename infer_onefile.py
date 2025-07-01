import argparse, json, os, tqdm, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROMPT_SYSTEM="You are a helpful AI assistant. 당신은 한국어 어문 규범 전문가입니다. 질문을 읽고 올바른 답과 이유를 제시하세요."
INST={
    "선택형":"[지침] 보기 중 올바른 표현을 골라 “○○가 옳다. 이유: …”로 답하십시오.",
    "교정형":"[지침] 틀린 부분을 고친 뒤 “○○가 옳다. 이유: …”로 답하십시오.",
    "선다형":"[지침] 가장 적절한 보기 번호만 숫자로 답하십시오.",
    "단답형":"[지침] 두 단어 이내로 간단히 답하십시오.",
    "서술형":"[지침] 완전한 문장으로 서술하십시오."
}

with open("hf_token.txt", "r") as f:
    hf_token = f.readline().strip()

#hf_token = "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"

class TestSet(Dataset):
    def __init__(self,path,tk):
        self.raw=json.load(open(path,encoding="utf-8"))
        self.tk=tk
        self.enc=[]
        for ex in self.raw:
            qt=ex["input"]["question_type"]
            q=ex["input"]["question"]
            chat=[{"role":"system","content":PROMPT_SYSTEM},
                  {"role":"user","content":f"{INST.get(qt,'')}\n\n[질문]\n{q}"}]
            self.enc.append(
                tk.apply_chat_template(chat,add_generation_prompt=True,return_tensors="pt",enable_thinking=False)[0]
            )
    def __len__(self):return len(self.enc)
    def __getitem__(self,idx):return self.enc[idx]

class Collator:
    def __init__(self,tk):self.tk=tk
    def __call__(self,batch):
        ids=torch.nn.utils.rnn.pad_sequence(batch,batch_first=True,padding_value=self.tk.pad_token_id)
        return{"input_ids":ids,"attention_mask":ids.ne(self.tk.pad_token_id)}

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True)
ap.add_argument("--output",required=True)
ap.add_argument("--model_id",required=True)
ap.add_argument("--tokenizer")
ap.add_argument("--device",default="cuda")
args=ap.parse_args()

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
mem={0:"13GiB","cpu":"32GiB"}
model_kwargs=dict(device_map="auto",quantization_config=bnb,max_memory=mem)
if hf_token:model_kwargs["use_auth_token"]=hf_token
model=AutoModelForCausalLM.from_pretrained(args.model_id,**model_kwargs)
model.eval()

tok_id=args.tokenizer or args.model_id
tk_kwargs={"use_auth_token":hf_token} if hf_token else {}
tk=AutoTokenizer.from_pretrained(tok_id,**tk_kwargs)
tk.pad_token=tk.eos_token
terminators=[tk.eos_token_id,tk.convert_tokens_to_ids("<|eot_id|>") or tk.convert_tokens_to_ids("<|endoftext|>")]

ds=TestSet(args.input,tk)
dl=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=Collator(tk))
out=json.load(open(args.input,encoding="utf-8"))
idx=0
with torch.no_grad():
    for batch in tqdm.tqdm(dl,total=len(dl)):
        batch={k:v.to(args.device) for k,v in batch.items()}
        gen=model.generate(**batch,max_new_tokens=200,eos_token_id=terminators,
                           pad_token_id=tk.eos_token_id,temperature=0.3,top_p=0.9,repetition_penalty=1.05)
        for b,g in zip(batch["input_ids"],gen):
            txt=tk.decode(g[b.shape[-1]:],skip_special_tokens=True).strip()
            for p in("답변:","답:","Answer:","answer:"):
                if txt.lower().startswith(p.lower()):txt=txt[len(p):].strip()
            if not txt.startswith('"'):txt='"'+txt
            if not txt.endswith('"'):txt=txt+'"'
            out[idx]["output"]={"answer":txt}
            idx+=1
with open(args.output,"w",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2)
print("Saved →",os.path.abspath(args.output))
