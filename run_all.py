import os, sys, json, textwrap, re, torch, pdfplumber
from pathlib import Path
from tqdm.auto import tqdm
import retriever, generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_SDP_BACKEND"] = "math"

PDF_PATH   = Path("Reference_Document.pdf")
REF_TXT    = Path("reference.txt")
TEST_JSON  = Path("data/korean_language_rag_V1.0_test.json")
SUBMISSION = Path("submission.json")

def build_reference():
    lines = []
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        for page in pdf.pages:
            txt = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if txt:
                lines.extend(textwrap.wrap(txt, 160))
    REF_TXT.write_text("\n".join(lines), encoding="utf-8")

def load_reference():
    if not REF_TXT.exists():
        if not PDF_PATH.exists():
            sys.exit(f"❌ {PDF_PATH} not found")
        build_reference()
    return REF_TXT.read_text(encoding="utf-8").splitlines()

def main():
    ref_lines = load_reference()
    ret = retriever.Retriever()
    ret.add_corpus(ref_lines)
    ret.build()

    items = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    gen = generator.Generator()

    out = []
    for it in tqdm(items, desc="inference"):
        q   = it["input"]["question"]
        ctx = [p for p, _ in ret.search(q, k=3)]
        with torch.no_grad():
            ans = gen.generate(q, ctx)
        out.append({"id": it["id"], "input": it["input"], "output": {"answer": ans}})

    SUBMISSION.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ saved → {SUBMISSION.resolve()}")

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()