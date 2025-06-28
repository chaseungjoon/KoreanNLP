import json, re, sys, difflib
refs = json.load(open('data/korean_language_rag_V1.0_dev.json'))
pred = json.load(open('submission.json'))
ref_map = {x['id']: x['output']['answer'] for x in refs}

EM = 0
for item in pred:
    gold = ref_map.get(item['id'])
    if not gold:
        continue
    if gold.strip() == item['output']['answer'].strip():
        EM += 1
print('예상 Exact Match :', EM, '/', len(pred))