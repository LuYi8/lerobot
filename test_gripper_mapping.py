import torch

# ========== 1:1 复刻你代码里的参数和逻辑 ==========
GRIPPER_DIM = 3
OPEN_THRESH = 1.2   # 大于该值 → 张开标签 0
CLOSE_THRESH = 0.8  # 小于该值 → 闭合标签 1

# 反向映射：标签 → 硬件开度
def label_to_openness(label: float) -> float:
    if label == 0:
        return 2.0  # 张开
    else:
        return 0.0  # 闭合

# 正向映射：硬件开度 → 标签（带滞回）
def openness_to_label(openness: float, last_label: float) -> tuple[float, float]:
    if openness > OPEN_THRESH:
        label = 0.0
    elif openness < CLOSE_THRESH:
        label = 1.0
    else:
        # 中间区间保持上一状态
        label = last_label
    return label, label

# ========== 测试用例 ==========
if __name__ == "__main__":
    print("=== 1. 边界值映射测试 ===")
    test_cases = [
        (2.0, 0.0, "完全张开"),
        (1.5, 0.0, "大于张开阈值"),
        (1.0, 0.0, "中间区间(从张开过来)"),
        (0.5, 1.0, "小于闭合阈值"),
        (0.0, 1.0, "完全闭合"),
    ]
    
    last_label = 0.0  # 初始默认张开
    all_pass = True
    for openness, expect_label, desc in test_cases:
        label, last_label = openness_to_label(openness, last_label)
        # 双向一致性校验：标签转回开度，看是否和预期一致
        recovered_openness = label_to_openness(label)
        passed = label == expect_label
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        print(f"{status} {desc}: 原值={openness:.2f}, 标签={label}, 转回开度={recovered_openness:.2f}, 预期标签={expect_label}")

    print("\n=== 2. 滞回逻辑测试 ===")
    # 场景：从闭合状态慢慢开到1.0（中间区间），应保持闭合标签1
    last_label = 1.0  # 上一状态是闭合
    label, last_label = openness_to_label(1.0, last_label)
    passed = label == 1.0
    all_pass = all_pass and passed
    print(f"{'✅' if passed else '❌'} 从闭合到中间值1.0: 标签={label}, 预期=1.0 (保持闭合)")

    # 场景：从张开状态慢慢关到1.0（中间区间），应保持张开标签0
    last_label = 0.0  # 上一状态是张开
    label, last_label = openness_to_label(1.0, last_label)
    passed = label == 0.0
    all_pass = all_pass and passed
    print(f"{'✅' if passed else '❌'} 从张开到中间值1.0: 标签={label}, 预期=0.0 (保持张开)")

    print("\n=== 3. 跨episode重置测试 ===")
    # 模拟上一个episode最后是闭合状态
    last_label = 1.0
    # 新episode重置为0.0（对应你代码里 episode 结束的重置逻辑）
    last_label = 0.0
    label, _ = openness_to_label(1.0, last_label)
    passed = label == 0.0
    all_pass = all_pass and passed
    print(f"{'✅' if passed else '❌'} 新episode初始中间值1.0: 标签={label}, 预期=0.0 (重置为张开)")

    print("\n" + "="*30)
    print(f"全部测试通过: {all_pass}")
