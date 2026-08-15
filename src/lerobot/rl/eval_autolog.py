"""评估结果自动落盘工具（本地实验工具链，不属于上游；供 eval_simple.py 调用）。

评估结束后把成功率与每集明细按 gym-hil/实验记录_整体.md §4 约定写入
归属的 checkpoint 文件夹：
  - 实验记录_<变体>_<超参段>.md   追加写入，不覆盖已有手写内容
  - eval_results.jsonl           每评估一行，机器可读（供多 seed 统计）

本模块零 torch / lerobot 依赖（纯标准库），可独立单元测试。

成功判据（在 eval_simple.py 主循环中收集，此处只记录口径）：
环境每步 info["succeed"]（panda_pick env 每步计算）。不用 done 判成功——
基础环境 terminated = success or exceeded_bounds（出界也终止），HIL wrapper
还把 truncated 并入 terminated（hil_wrappers.py:246）。
"""
import json
import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

_HP_FIELDS = [
    "use_recurrent",
    "critic_use_recurrent",
    "recurrent_hidden_size",
    "recurrent_num_layers",
    "sequence_length",
    "tau",
    "offline_capacity",
    "online_capacity",
    "source",
]


def _get(obj, key, default=None):
    """统一取 dict / dataclass 字段（CLI cfg 与 train_config.json 两种来源）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def collect_hyperparams(checkpoint_dir, cfg=None) -> dict:
    """收集实验记录命名所需超参，三级回退：train_config.json > CLI cfg > None。

    train_config.json 是训练时的权威配置（每个 checkpoint 都有）；缺失/损坏时
    回退本次评估的 CLI cfg（actor_hil_env.json 亦含全部字段）；再取不到的分量
    保持 None（文件名以 unk 占位），绝不因命名失败中断评估。
    """
    hp = {k: None for k in _HP_FIELDS}
    hp["use_recurrent"] = False
    hp["critic_use_recurrent"] = False

    tc_path = os.path.join(checkpoint_dir, "train_config.json")
    try:
        with open(tc_path, encoding="utf-8") as f:
            tc = json.load(f)
        alg = tc.get("algorithm") or {}
        pol = tc.get("policy") or {}
        pk = pol.get("policy_kwargs") or {}
        hp.update(
            use_recurrent=bool(pk.get("use_recurrent", False)),
            critic_use_recurrent=bool(alg.get("critic_use_recurrent", False)),
            recurrent_hidden_size=pk.get("recurrent_hidden_size"),
            recurrent_num_layers=pk.get("recurrent_num_layers"),
            sequence_length=alg.get("sequence_length"),
            tau=alg.get("critic_target_update_weight"),
            offline_capacity=pol.get("offline_buffer_capacity"),
            online_capacity=pol.get("online_buffer_capacity"),
            source="train_config.json",
        )
        return hp
    except (OSError, ValueError):
        pass

    if cfg is not None:
        try:
            alg = _get(cfg, "algorithm") or {}
            pol = _get(cfg, "policy") or {}
            pk = _get(pol, "policy_kwargs") or {}
            hp.update(
                use_recurrent=bool(_get(pk, "use_recurrent", False)),
                critic_use_recurrent=bool(_get(alg, "critic_use_recurrent", False)),
                recurrent_hidden_size=_get(pk, "recurrent_hidden_size"),
                recurrent_num_layers=_get(pk, "recurrent_num_layers"),
                sequence_length=_get(alg, "sequence_length"),
                tau=_get(alg, "critic_target_update_weight"),
                offline_capacity=_get(pol, "offline_buffer_capacity"),
                online_capacity=_get(pol, "online_buffer_capacity"),
                source="cli_cfg",
            )
        except Exception:
            pass
    return hp


def build_record_filename(hp: dict) -> str:
    """按超参生成实验记录文件名，如 实验记录_RSAC_h256_seq8_tau0005_off8500_on1500.md。

    变体：纯SAC / RSAC / 仅criticGRU / 双关；
    超参段（_ 连接，顺序固定）：
      - h{N}          仅循环变体（use_recurrent / critic_use_recurrent）出现
      - layers{N}     仅 != 1 时出现（默认 1 省略，与既有样例一致）
      - seq{N}        sequence_length > 1 时无论变体都带（>1 即序列采样开启，
                      独立于 GRU 的训练语义，防止纯SAC 不同 seq 撞名）
      - tau{去点}     恒带（τ 已在 0.05/0.005 间变化，不入名会同变体撞名）
      - off{N}/on{N}  恒带
    """
    seq = hp.get("sequence_length")
    seq_seg = f"seq{int(seq)}" if (seq or 0) > 1 else ""
    hidden = hp.get("recurrent_hidden_size")
    layers = hp.get("recurrent_num_layers")
    use_rec = bool(hp.get("use_recurrent"))
    use_crit = bool(hp.get("critic_use_recurrent"))

    detail = []
    if use_rec or use_crit:
        if hidden:
            detail.append(f"h{int(hidden)}")
        if layers and int(layers) != 1:
            detail.append(f"layers{int(layers)}")
    if seq_seg:
        detail.append(seq_seg)
    detail_seg = "_".join(detail)

    if use_rec and use_crit:
        base = "双关"
    elif use_rec:
        base = "RSAC"
    elif use_crit:
        base = "仅criticGRU"
    else:
        base = "纯SAC"
    variant = f"{base}_{detail_seg}" if detail_seg else base

    tau = hp.get("tau")
    tau_seg = f"tau{str(tau).replace('.', '')}" if tau is not None else "tau_unk"
    off = hp.get("offline_capacity")
    off_seg = f"off{int(off)}" if off is not None else "off_unk"
    on = hp.get("online_capacity")
    on_seg = f"on{int(on)}" if on is not None else "on_unk"
    return f"实验记录_{variant}_{tau_seg}_{off_seg}_{on_seg}.md"


def dump_eval_record(out_dir: str, hp: dict, episodes: list, command: str | None = None) -> None:
    """把评估结果追加写入 checkpoint 文件夹（md + jsonl），失败仅告警不抛。

    episodes: 每集 (reward, steps, success, intervention_steps) 的元组/列表；
    时间戳取一次，md 与 jsonl 共用；写盘在 os._exit(0) 前调用且显式 flush。
    """
    out_dir = str(out_dir)  # draccus 路径字段可能是 Path 对象，json 需要 str
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if command is None:
        command = "python -m lerobot.rl.eval_simple " + " ".join(sys.argv[1:])

    # 规范化每集明细（防御：元组长度不足/类型不规整时补默认值）
    rows = []
    for ep in episodes:
        r, s, ok, iv = (list(ep) + [0.0, 0, False, 0])[:4]
        rows.append(
            {
                "reward": float(r),
                "steps": int(s),
                "success": bool(ok),
                "intervention_steps": int(iv),
            }
        )
    n = len(rows)
    n_success = sum(1 for row in rows if row["success"])
    rate = n_success / n if n else 0.0
    avg_reward = sum(row["reward"] for row in rows) / n if n else 0.0
    avg_steps = sum(row["steps"] for row in rows) / n if n else 0.0
    intv_steps = sum(row["intervention_steps"] for row in rows)
    total_steps = sum(row["steps"] for row in rows)
    intv_rate = intv_steps / total_steps if total_steps else 0.0

    # ---- md：追加（文件已存在——含手写记录——原文保留，只在文末加节） ----
    try:
        md_path = os.path.join(out_dir, build_record_filename(hp))
        existed = os.path.exists(md_path)
        lines = []
        if not existed:
            lines.append("# 实验记录（自动落盘）\n")
            lines.append(f"- 归属 checkpoint：`{out_dir}`\n")
            lines.append('- 成功判据：环境每步 info["succeed"]（不用 done——出界/超时也会 done）\n')
            lines.append(f"- 首条自动评估：{timestamp}\n")
        lines.append(f"\n---\n\n## 自动评估 {timestamp}\n\n")
        lines.append(f"- 命令：`{command}`\n")
        lines.append(f"- n_episodes：{n}；成功率：**{n_success}/{n} = {rate:.1%}**\n")
        lines.append(
            f"- 平均 reward：{avg_reward:.3f}；平均 steps：{avg_steps:.1f}；"
            f"干预 {intv_steps}/{total_steps} 步（{intv_rate:.1%}）\n"
        )
        hp_line = (
            f"τ={hp.get('tau')}，seq={hp.get('sequence_length')}，"
            f"use_recurrent={bool(hp.get('use_recurrent'))}，"
            f"critic_use_recurrent={bool(hp.get('critic_use_recurrent'))}，"
            f"offline={hp.get('offline_capacity')}，online={hp.get('online_capacity')}"
        )
        if hp.get("use_recurrent"):
            hp_line += (
                f"，hidden={hp.get('recurrent_hidden_size')}，layers={hp.get('recurrent_num_layers')}"
            )
        lines.append(f"- 超参（来源：{hp.get('source')}）：{hp_line}\n")
        lines.append(f"- 每集明细（reward/steps/成功/干预步）：\n")
        for i, row in enumerate(rows):
            detail = f"  - ep{i + 1}: r={row['reward']:.2f}, s={row['steps']}, {'✓' if row['success'] else '✗'}"
            if row["intervention_steps"]:
                detail += f"（干预 {row['intervention_steps']} 步）"
            lines.append(detail + "\n")
        lines.append(f"\n（结论：待人工回填）\n")
        with open(md_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[eval] 实验记录已追加: {md_path}")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"实验记录 md 落盘失败（评估结果仍在上方终端输出）: {e}")

    # ---- jsonl：机器可读，每评估一行 ----
    try:
        record = {
            "timestamp": timestamp,
            "command": command,
            "checkpoint": out_dir,
            "n_episodes": n,
            "success_count": n_success,
            "success_rate": rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "intervention_steps": intv_steps,
            "intervention_rate": intv_rate,
            "success_criterion": "env info['succeed']",
            "hyperparams": hp,
            "per_episode": rows,
        }
        jsonl_path = os.path.join(out_dir, "eval_results.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[eval] 机器可读结果已追加: {jsonl_path}")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"eval_results.jsonl 落盘失败: {e}")
