import pandas as pd
import glob

dataset_root = "/home/embody/lerobot/gym-hil/dataset"
parquet_files = glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True)
df = pd.read_parquet(parquet_files[0])

# 取第0个完整episode
ep0 = df[df["episode_index"] == 0].reset_index(drop=True)
gripper_vals = ep0["action"].apply(lambda x: x[3]).values

# 代入你的离散化规则
threshold = 0.5
labels = (gripper_vals <= threshold).astype(float)

print("帧号\t硬件值\t标签\t物理预判")
print("-" * 40)
for i in range(len(gripper_vals)):
    state = "张开" if gripper_vals[i] < 0.5 else "闭合"
    print(f"{i}\t{gripper_vals[i]:.1f}\t{labels[i]:.0f}\t{state}")

# 统计标签切换次数
switch_count = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
print(f"\n整个episode标签切换次数: {switch_count}")
