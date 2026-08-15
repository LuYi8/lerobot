#!/usr/bin/env python
"""沙箱验证 eval_autolog.py 自动落盘（纯函数级，不依赖 GPU / 真环境）。

必须用仓库实际环境运行（Python >= 3.12，lero6）：
    ~/miniconda3/envs/lero6/bin/python gym-hil/tests/verify_eval_autolog.py

覆盖五项：
1. 文件名生成：四变体 × seq>1/1 × τ × off/on × unk 回退（含纯SAC+seq8 回归）
2. 超参三级回退：train_config.json > 损坏回退 CLI cfg > 全缺失不崩
3. md 追加语义：手写内容保留、新节按序追加
4. jsonl 逐行追加可解析
5. 容错：只读目录 / hp 缺键（.get 防御）/ 干预口径
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.rl.eval_autolog import (  # noqa: E402
    build_record_filename,
    collect_hyperparams,
    dump_eval_record,
)


def _make_cfg(**overrides):
    """构造最小 TrainRLServerPipelineConfig mock（只覆盖被测字段）。"""
    cfg = MagicMock()
    cfg.policy.pretrained_path = overrides.get("pretrained_path")
    cfg.policy.policy_kwargs = overrides.get("policy_kwargs", MagicMock())
    cfg.policy.offline_buffer_capacity = overrides.get("offline_buffer_capacity")
    cfg.policy.online_buffer_capacity = overrides.get("online_buffer_capacity")
    cfg.algorithm = overrides.get("algorithm", MagicMock())
    return cfg


# ---------- 1. 文件名生成 ----------

def test_filename_generation():
    cases = [
        # (label, hp, expected)
        (
            "纯SAC τ=0.005（无循环、seq1 不带段）",
            dict(use_recurrent=False, critic_use_recurrent=False, sequence_length=1,
                 tau=0.005, offline_capacity=8500, online_capacity=1500),
            "实验记录_纯SAC_tau0005_off8500_on1500.md",
        ),
        (
            "纯SAC + seq8（序列采样开启也入名，防撞名回归）",
            dict(use_recurrent=False, critic_use_recurrent=False, sequence_length=8,
                 tau=0.005, offline_capacity=8500, online_capacity=1500),
            "实验记录_纯SAC_seq8_tau0005_off8500_on1500.md",
        ),
        (
            "R-SAC h256 layers1 seq8 τ=0.005（与手写样例尾部一致）",
            dict(use_recurrent=True, critic_use_recurrent=False,
                 recurrent_hidden_size=256, recurrent_num_layers=1,
                 sequence_length=8, tau=0.005,
                 offline_capacity=8500, online_capacity=1500),
            "实验记录_RSAC_h256_seq8_tau0005_off8500_on1500.md",
        ),
        (
            "R-SAC layers=2 才带 layers 段",
            dict(use_recurrent=True, critic_use_recurrent=False,
                 recurrent_hidden_size=256, recurrent_num_layers=2,
                 sequence_length=8, tau=0.005,
                 offline_capacity=8500, online_capacity=1500),
            "实验记录_RSAC_h256_layers2_seq8_tau0005_off8500_on1500.md",
        ),
        (
            "仅criticGRU seq8 τ=0.005",
            dict(use_recurrent=False, critic_use_recurrent=True,
                 sequence_length=8, tau=0.005,
                 offline_capacity=8500, online_capacity=4000),
            "实验记录_仅criticGRU_seq8_tau0005_off8500_on4000.md",
        ),
        (
            "双关 h256 seq8 τ=0.005（§4 示例同款）",
            dict(use_recurrent=True, critic_use_recurrent=True,
                 recurrent_hidden_size=256, recurrent_num_layers=1,
                 sequence_length=8, tau=0.005,
                 offline_capacity=8500, online_capacity=4000),
            "实验记录_双关_h256_seq8_tau0005_off8500_on4000.md",
        ),
        (
            "纯SAC τ=0.05（历史默认）",
            dict(use_recurrent=False, critic_use_recurrent=False, tau=0.05,
                 offline_capacity=2000, online_capacity=8000),
            "实验记录_纯SAC_tau005_off2000_on8000.md",
        ),
        (
            "全 unk 回退",
            dict(use_recurrent=False, critic_use_recurrent=False,
                 tau=None, offline_capacity=None, online_capacity=None),
            "实验记录_纯SAC_tau_unk_off_unk_on_unk.md",
        ),
    ]
    for label, hp, expected in cases:
        got = build_record_filename(hp)
        assert got == expected, f"[{label}] 期望 {expected!r}, 得到 {got!r}"
    print("[PASS] 文件名生成（8 个组合，含 seq 回归与 unk 回退）")


# ---------- 2. 超参三级回退 ----------

def test_collect_hyperparams(tmp_path):
    tc = {
        "algorithm": {
            "critic_target_update_weight": 0.005,
            "sequence_length": 8,
            "critic_use_recurrent": False,
        },
        "policy": {
            "policy_kwargs": {
                "use_recurrent": True,
                "recurrent_hidden_size": 256,
                "recurrent_num_layers": 1,
            },
            "offline_buffer_capacity": 8500,
            "online_buffer_capacity": 1500,
        },
    }

    # 1) 有 train_config.json → 用它
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "train_config.json").write_text(json.dumps(tc), encoding="utf-8")
    cfg = _make_cfg(pretrained_path=str(ckpt))
    hp = collect_hyperparams(str(ckpt), cfg)
    assert hp["source"] == "train_config.json"
    assert hp["use_recurrent"] is True and hp["tau"] == 0.005
    assert hp["sequence_length"] == 8 and hp["offline_capacity"] == 8500
    print("[PASS] 超参来源 = train_config.json")

    # 2) train_config.json 损坏 → 回退 CLI cfg
    (ckpt / "train_config.json").write_text("{bad json!!!", encoding="utf-8")
    cfg = _make_cfg(
        pretrained_path=str(ckpt),
        algorithm=MagicMock(critic_target_update_weight=0.05, sequence_length=1,
                            critic_use_recurrent=False),
        policy_kwargs=MagicMock(use_recurrent=False),
        offline_buffer_capacity=2000,
        online_buffer_capacity=8000,
    )
    hp = collect_hyperparams(str(ckpt), cfg)
    assert hp["source"] == "cli_cfg"
    assert hp["tau"] == 0.05 and hp["use_recurrent"] is False
    print("[PASS] 超参来源 = cli_cfg（train_config.json 损坏回退）")

    # 3) 无 json + CLI cfg 字段全缺 → 不崩、值 None、source 保持 unk
    (ckpt / "train_config.json").unlink()
    cfg = _make_cfg(pretrained_path=str(ckpt))
    cfg.policy.policy_kwargs = type("Empty", (), {})()
    cfg.algorithm = type("Empty", (), {})()
    hp = collect_hyperparams(str(ckpt), cfg)
    assert hp["tau"] is None and hp["offline_capacity"] is None
    print("[PASS] 超参回退无崩溃（全字段缺失，unk 占位）")


# ---------- 3/4. md 追加 + jsonl ----------

def test_dump_append_and_jsonl(tmp_path):
    ckpt = tmp_path / "ckpt2"
    ckpt.mkdir()
    hp = dict(
        use_recurrent=True, critic_use_recurrent=False,
        recurrent_hidden_size=256, recurrent_num_layers=1,
        sequence_length=8, tau=0.005,
        offline_capacity=8500, online_capacity=1500,
        source="test",
    )
    # 预置手写 md（模拟 EXP-007 人工记录）
    md_file = ckpt / "实验记录_RSAC_h256_seq8_tau0005_off8500_on1500.md"
    md_file.write_text(
        "# 实验记录：无 HIL R-SAC（EXP-007）\n"
        "- **日期**：2026-08-14\n"
        "- 成功率：**100%**\n"
        "（手工写的结论）\n",
        encoding="utf-8",
    )

    # 两次评估：ep1 成功无干预、ep2 失败含 3 步干预；第二次全成功
    dump_eval_record(
        str(ckpt), hp,
        [(1.0, 42, True, 0), (0.8, 80, False, 3)],
        command="python -m lerobot.rl.eval_simple --eval.n_episodes 2",
    )
    dump_eval_record(
        str(ckpt), hp,
        [(1.0, 38, True, 0), (1.0, 40, True, 0)],
        command="python -m lerobot.rl.eval_simple --eval.n_episodes 2",
    )

    # md：手写原文保留 + 两节按序追加
    content = md_file.read_text(encoding="utf-8")
    assert content.startswith("# 实验记录：无 HIL R-SAC（EXP-007）"), "手写原文被覆盖！"
    assert content.count("## 自动评估") == 2, f"期望 2 节，得到 {content.count('## 自动评估')} 节"
    assert "r=1.00, s=42" in content and "r=0.80, s=80" in content
    assert "（干预 3 步）" in content, "干预步明细缺失"
    assert content.index("## 自动评估") < content.rindex("## 自动评估"), "节序错乱"
    print("[PASS] md 追加：手写原文保留 + 两节按序追加 + 干预明细")

    # jsonl：两行可解析、字段完整
    jsonl_file = ckpt / "eval_results.jsonl"
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"期望 2 行 jsonl，得到 {len(lines)} 行"
    r1, r2 = (json.loads(l) for l in lines)
    assert r1["success_count"] == 1 and r1["success_rate"] == 0.5
    assert r1["intervention_steps"] == 3
    assert r1["intervention_rate"] == 3 / (42 + 80)
    assert r2["success_count"] == 2 and r2["success_rate"] == 1.0
    assert r1["per_episode"][1] == {"reward": 0.8, "steps": 80, "success": False, "intervention_steps": 3}
    assert r1["hyperparams"]["source"] == "test"
    assert r1["success_criterion"] == "env info['succeed']"
    print("[PASS] jsonl 两行解析正确（成功率/干预率/per_episode/hyperparams）")


# ---------- 5. 容错 ----------

def test_dump_failure_no_crash(tmp_path):
    # 只读目录：md/jsonl 落盘失败仅告警不抛
    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(0o444)
    hp = dict(use_recurrent=False, critic_use_recurrent=False, tau=0.005,
              offline_capacity=8500, online_capacity=1500, source="test")
    try:
        dump_eval_record(str(ro), hp, [(1.0, 50, True, 0)])
        print("[PASS] 只读目录落盘不抛异常")
    finally:
        ro.chmod(0o755)

    # hp 缺键（模拟外部传入不完整 dict）：.get 防御，md 仍可生成
    ckpt = tmp_path / "ckpt3"
    ckpt.mkdir()
    dump_eval_record(str(ckpt), {"tau": 0.005}, [(1.0, 10, True, 0)])
    md_files = [p.name for p in ckpt.iterdir() if p.suffix == ".md"]
    assert md_files, "hp 缺键时仍应生成 md"
    print(f"[PASS] hp 缺键防御（生成 {md_files[0]}）")


if __name__ == "__main__":
    print("=" * 60)
    print("verify_eval_autolog：eval_autolog 自动落盘沙箱验证")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_filename_generation()
        test_collect_hyperparams(tmp_path)
        test_dump_append_and_jsonl(tmp_path)
        test_dump_failure_no_crash(tmp_path)
    print("\n" + "=" * 60)
    print("全部 5 项 PASS")
    print("=" * 60)
