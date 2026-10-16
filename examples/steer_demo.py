"""把 zhsteer 接进现有 transformers 管线的最小示例（需要 CUDA GPU）。

先用 `zhsteer extract --model qwen3-4b` 生成方向包，然后：

    uv run python examples/steer_demo.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from zhsteer import load, steer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pack = load("packs/qwen3-4b.pack")  # checksum + model tag verified on load

inputs = tokenizer("写一段产品介绍", return_tensors="pt").to(model.device)
with steer(model, pack, strength=0.9):  # 消融强度：1.0 完全移除，0 关闭
    output = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.decode(output[0], skip_special_tokens=True))
