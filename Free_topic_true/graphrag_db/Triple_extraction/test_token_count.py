"""
测试 token 计数功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from Free_topic_true.graphrag_db.Triple_extraction.TE_read_json import count_tokens, load_records

# 加载数据
records = load_records("./act-1997-078_chunked_section.json")

print(f"Analyzing {len(records)} records...\n")
print(f"{'Record':<10} {'Chars':<10} {'Tokens':<10}")
print("-" * 35)

total_chars = 0
total_tokens = 0

for idx, rec in enumerate(records[:5], 1):  # 只显示前 5 条
    text = rec.get("text", "")
    chars = len(text)
    tokens = count_tokens(text)
    
    total_chars += chars
    total_tokens += tokens
    
    print(f"{idx:<10} {chars:<10} {tokens:<10}")

print("-" * 35)
print(f"{'Total':<10} {total_chars:<10} {total_tokens:<10}")
print(f"\nAverage tokens per char: {total_tokens / total_chars:.3f}")
