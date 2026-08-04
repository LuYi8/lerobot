from safetensors.torch import load_file

ckpt_path = "/home/embody/lerobot/gym-hil/output/checkpoints/last/pretrained_model/model.safetensors"
state_dict = load_file(ckpt_path)

# 1. 拆分有效actor权重、丢弃冗余encoder_actor/encoder_critic
actor_clean = {}
redundant = []
for full_key, weight in state_dict.items():
    if full_key.startswith("actor."):
        # 剥离 actor. 前缀，和policy内部命名对齐
        new_key = full_key.removeprefix("actor.")
        actor_clean[new_key] = weight
    elif full_key.startswith("encoder_actor.") or full_key.startswith("encoder_critic."):
        redundant.append(full_key)

print("===== 权重分析结果 =====")
print(f"总权重条目: {len(state_dict)}")
print(f"有效Actor权重(剥离actor.后): {len(actor_clean)}")
print(f"冗余旧编码器权重(会产生Unexpected警告): {len(redundant)}")
print("\n【冗余key示例（日志警告来源）】")
for k in redundant[:5]:
    print("  ", k)

print("\n【剥离前缀后的前10个有效权重名（模型能匹配的）】")
keys_list = list(actor_clean.keys())
for k in keys_list[:10]:
    print("  ", k)

# 统计视觉编码器、GRU数量
img_encoder_cnt = sum(1 for k in actor_clean if "image_encoder" in k)
gru_cnt = sum(1 for k in actor_clean if "gru" in k)
print(f"\n图像编码器ResNet参数数量：{img_encoder_cnt}")
print(f"GRU层参数数量：{gru_cnt}")
