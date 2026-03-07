# %%
import os
import sys
import copy
import time
import torch
import pickle
import pprint
import logging
import argparse
import transformers
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
import json
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

transformers.logging.set_verbosity_error()

sys.path.insert(1, os.path.dirname(os.getcwd()))
from plotting import (
    visualize_acc_rate_over_time,
)
from utils import (
    Colors, is_interactive,
    populate_dataset, get_first_user_msg, 
    format_problem_and_options, format_drafter_name, get_proposal_str, get_output_tokens,
    get_output_dir,
    print_sd_trajectory,
)

def build_states_for_math(args, target_tokenizer, num_questions=500):
    # assume populate_dataset(args) already done and args.dataset ready
    states = []
    for problem_id in range(num_questions):
        raw_data = format_problem_and_options(args, problem_id)
        messages = [{"role": "user", "content": get_first_user_msg(args, raw_data)}]
        text = target_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        orig = target_tokenizer([text], return_tensors="pt")
        orig_input_ids = orig["input_ids"][0].tolist()

        states.append({
            "problem_id": problem_id,
            "raw_data": raw_data,              # optional (để debug/print)
            "orig_input_ids": orig_input_ids,  # gửi cho worker
            "current_token_ids": [],
            "prev_prefill_output": None,
            "done": False,
            "stats_each_round": [],
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "num_speculation_rounds": 0,
        })
    return states

def split_indices(num_questions, num_drafters=3):
    buckets = [[] for _ in range(num_drafters)]
    for pid in range(num_questions):
        buckets[pid % num_drafters].append(pid)
    return buckets

def verify_batch_target(target_model, tokenizer, batch_items, num_target_tokens):
    device = torch.device("cuda:0")

    seqs = []
    masks = []
    meta = []

    for item in batch_items:
        # item includes: orig_input_ids(list), current_token_ids(list), draft(list)
        orig_ids = item["orig_input_ids"]
        cur = item["current_token_ids"]
        draft = item["draft"]

        prompt_len = len(orig_ids)
        prefix_len = len(cur)

        full = orig_ids + cur + draft
        full_t = torch.tensor(full, device=device, dtype=torch.long)
        attn = torch.ones_like(full_t, dtype=torch.long)

        # slice logits for draft verification
        start = prompt_len + prefix_len - 1
        end = start + len(draft)

        seqs.append(full_t)
        masks.append(attn)
        meta.append((start, end, draft, prefix_len))

    # pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batch_input = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_mask = pad_sequence(masks, batch_first=True, padding_value=0)

    # time verify on GPU0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_evt.record()

    with torch.inference_mode():
        out = target_model(input_ids=batch_input, attention_mask=batch_mask)
        logits = out.logits  # [B, T, V]

    end_evt.record()
    torch.cuda.synchronize()
    verify_time_ms = start_evt.elapsed_time(end_evt)

    # accept/reject per sample
    updates = []
    for b, (start, end, draft, prefix_len) in enumerate(meta):
        verify_logits = logits[b, start:end, :]
        accepted_len = 0
        bonus_token = None

        for i in range(len(draft)):
            pred = torch.argmax(verify_logits[i], dim=-1).item()
            if draft[i] == pred:
                accepted_len += 1
            else:
                final_token = pred
                break
        else:
            final_token = torch.argmax(logits[b, -1, :], dim=-1).item()
            bonus_token = final_token

        tokens_to_append = draft[:accepted_len] + [final_token]
        updates.append({
            "accepted_len": accepted_len,
            "final_token": final_token,
            "bonus_token": bonus_token,
            "tokens_to_append": tokens_to_append,
        })

    return updates, verify_time_ms


# %%
def get_target_token_ids(model, tokenizer, messages, max_new_tokens):
    """Get the target series of token IDs for the given messages.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    num_input_tokens = model_inputs.input_ids.shape[1]
    logging.debug(f"num_input_tokens {num_input_tokens}, first eight tokens: {model_inputs.input_ids[0, :8].tolist()}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # use greedy decoding, not sampling; overrides all below sampling params
        # temperature=0.0, top_p=1.0, top_k=0.0,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs


def get_next_n_tokens_ar(model, orig_model_inputs, token_ids_so_far, n):
    """Get the next n tokens from the model given the token IDs so far.
    """
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = model.generate(
        **new_model_inputs,
        max_new_tokens=n,
        do_sample=False,  # use greedy decoding, not sampling; overrides all below sampling params
        # temperature=0.0, top_p=1.0, top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()


def get_next_n_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
    """Get the next n tokens from the model given the token IDs so far.
    """
    # num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    input_ids = orig_model_inputs["input_ids"] if isinstance(orig_model_inputs, dict) else orig_model_inputs.input_ids
    num_tokens_in_prompt = input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    if args.disable_reusing_drafter_kvs:
        generated_ids, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency, prob because we are running a 1.5B model, which is memory-bound
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
        )
    else:
        generated_ids, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:spec_len]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, num_forward_passes, forward_pass_latencies


def get_next_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None,
                        lowconf_threshold=None,
                        max_spec_len=None,
                        incr_len=None,
                        last_round_rejected=None,
    ):
    """Get the next few tokens from the model given the token IDs so far.
    Difference is that the dLLM drafter itself determines how many tokens to output based on model internal signals.
    """
    # num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    input_ids = orig_model_inputs["input_ids"] if isinstance(orig_model_inputs, dict) else orig_model_inputs.input_ids
    num_tokens_in_prompt = input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    if args.disable_reusing_drafter_kvs:
        generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
        )
    else:
        generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:actual_spec_len]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies


def construct_drafter_configs(args):
    drafter_configs = []
    if args.run_ar:  # AR Drafter
        drafter_configs.extend([("ar", None, "sf", None, None, None)] )
    if args.run_dllm_sf:  # Fast-dLLM Drafter
        drafter_configs.extend([("dllm", thr, "sf", None, None, None) for thr in args.drafter_thresholds])
    if not args.baseline_sweep:  # FailFast Drafter
        drafter_configs.extend([("dllm", thr, "df", lowconf_threshold, max_spec_len, incr_len) 
                                for thr in args.drafter_thresholds
                                for lowconf_threshold in args.sweep_lowconf_threshold
                                for max_spec_len in args.sweep_max_spec_len
                                for incr_len in args.sweep_incr_len
                                ])
    args.drafter_configs = drafter_configs

def drafter_worker(worker_id, gpu_id, args, in_q: mp.Queue, out_q: mp.Queue):
    """
    Drafter worker chạy trên 1 GPU (gpu_id). Giữ KV cache nội bộ theo problem_id.
    KHÔNG gửi prefill_output qua queue (tránh pickle lỗi transformers_modules).
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    dllm = AutoModelForCausalLM.from_pretrained(
        args.dllm_dir,
        torch_dtype="auto",
        device_map={"": gpu_id},
        trust_remote_code=True,
    ).eval()

    active_problem_id = None
    active_prefill = None  # giữ KV cho đúng 1 câu hiện tại

    prefill_cache = {}  # problem_id -> prefill_output (giữ nội bộ)

    while True:
        msg = in_q.get()
        if msg is None:
            break

        problem_id = msg["problem_id"]
        orig_input_ids = msg["orig_input_ids"]          # list[int]
        current_token_ids = msg["current_token_ids"]    # list[int]
        drafter_threshold = msg["drafter_threshold"]

        # dựng orig_model_inputs dạng dict (hàm get_next_n_tokens_dllm đã được vá để hiểu dict)
        orig_model_inputs = {
            "input_ids": torch.tensor([orig_input_ids], device=device, dtype=torch.long),
            "attention_mask": torch.ones((1, len(orig_input_ids)), device=device, dtype=torch.long),
        }

        if problem_id != active_problem_id:
            active_problem_id = problem_id
            active_prefill = None
            torch.cuda.empty_cache()  # optional: giúp trả lại reserved blocks

        prev_prefill = active_prefill

        spec_len = msg["spec_len"]              # quota từ verifier
        output_seqlen = msg.get("output_seqlen", 1 * args.block_size)
        small_block_size = msg.get("small_block_size", 8)

        # timing GPU
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()

        # baseline static-frequency dLLM
        draft, prefill_output, num_fwd, lat_list = get_next_n_tokens_dllm(
            dllm, args, orig_model_inputs, current_token_ids,
            spec_len=spec_len,
            output_seqlen=output_seqlen,
            small_block_size=small_block_size,
            threshold=drafter_threshold,
            is_drafter=True,
            prev_prefill_output=prev_prefill,
        )

        active_prefill = prefill_output

        end_evt.record()
        torch.cuda.synchronize()
        draft_time_ms = float(start_evt.elapsed_time(end_evt))

        # cập nhật cache nội bộ
        prefill_cache[problem_id] = prefill_output

        # gửi message “thuần” (picklable)
        out_q.put({
            "worker_id": int(worker_id),
            "problem_id": int(problem_id),
            "draft": draft,  # list[int]
            "num_forward_passes": int(num_fwd),
            "draft_time_ms": draft_time_ms,
        })

def optimizer(
    *,
    dispatched_workers,                 # list[int]
    round_acc,                          # dict[w] -> (accepted_len, spec_used)
    draft_gpu_ms,                       # dict[w] -> draft_time_ms (GPU time)
    verify_gpu_ms,                      # float (GPU time on verifier)
    ema_acc,                            # list[float], len=num_workers (in/out)
    ema_ms_per_tok,                     # list[float], len=num_workers (in/out)
    prev_next_spec_len,                 # list[int], len=num_workers (in)
    verifier_budget,                    # int: max total draft tokens to verify next round
    min_spec_len=1,
    max_spec_len=32,
    balance_alpha=1.0,                  # time_cap_ms = alpha * verify_gpu_ms
    ema_beta=0.2,                       # EMA smoothing
):
    """
    Tính next_spec_len[w] cho round kế tiếp dựa trên:
      - acceptance rate (EMA)
      - draft speed (EMA ms/token)
      - time_cap theo drafter chậm nhất (dựa trên verify_gpu_ms)
      - verifier_budget (tổng token verify tối đa 1 lần)
    Trả về: next_spec_len (dict w->len), và cập nhật ema_acc/ema_ms_per_tok tại chỗ.
    """
    # 1) Update EMA cho các worker active trong round
    for w in dispatched_workers:
        accepted_len, spec_used = round_acc[w]
        spec_used = max(1, int(spec_used))
        acc = float(accepted_len) / spec_used
        mspt = float(draft_gpu_ms[w]) / spec_used

        ema_acc[w] = (1.0 - ema_beta) * ema_acc[w] + ema_beta * acc
        ema_ms_per_tok[w] = (1.0 - ema_beta) * ema_ms_per_tok[w] + ema_beta * mspt

    # 2) Time cap: muốn draft stage ~ verify stage (để không chờ quá lâu)
    time_cap_ms = float(balance_alpha) * float(verify_gpu_ms)

    # 3) Score để phân bổ budget: acc cao + gen nhanh => score lớn
    cap_by_time = {}
    score = {}
    for w in dispatched_workers:
        mspt = max(1e-6, float(ema_ms_per_tok[w]))
        cap_by_time[w] = max(min_spec_len, int(time_cap_ms / mspt))
        score[w] = max(1e-3, float(ema_acc[w])) / mspt  # càng lớn càng tốt

    # 4) Allocate verifier budget B theo score
    B = int(verifier_budget)
    if B < min_spec_len * len(dispatched_workers):
        # budget quá thấp -> ép tối thiểu
        return {w: min_spec_len for w in dispatched_workers}

    score_sum = sum(score.values()) + 1e-9
    proposed = {}
    for w in dispatched_workers:
        alloc = int(B * score[w] / score_sum)
        proposed[w] = max(min_spec_len, min(max_spec_len, cap_by_time[w], alloc))

    # 5) Repair để đảm bảo sum <= B và không ai < min_spec_len
    #    Nếu sum quá lớn -> giảm dần từ worker score thấp
    def total():
        return sum(proposed.values())

    while total() > B:
        candidates = [w for w in dispatched_workers if proposed[w] > min_spec_len]
        if not candidates:
            break
        w_bad = min(candidates, key=lambda x: score[x])
        proposed[w_bad] -= 1

    # 6) Nếu sum quá nhỏ (do làm tròn) và còn budget -> cộng thêm theo score cao
    while total() < B:
        candidates = [w for w in dispatched_workers if proposed[w] < min(max_spec_len, cap_by_time[w])]
        if not candidates:
            break
        w_good = max(candidates, key=lambda x: score[x])
        proposed[w_good] += 1

    return proposed

def optimizer_pf_log(
    *,
    dispatched_workers,                 # list[int]
    round_acc,                          # dict[w] -> (accepted_len, spec_used)
    draft_gpu_ms,                       # dict[w] -> draft_time_ms (GPU time)
    verify_gpu_ms,                      # float (GPU time on verifier) last round
    ema_acc,                          # list[float], len=num_workers (in/out): EMA of alpha (match prob)
    ema_ms_per_tok,                     # list[float], len=num_workers (in/out): EMA of draft ms/token
    prev_next_spec_len,                 # list[int], len=num_workers (in) (optional, for warm-start)
    verifier_budget,                    # int: max total draft tokens to verify next round (B)
    min_spec_len=1,
    max_spec_len=32,
    balance_alpha=1.0,                  # time_cap_ms = balance_alpha * verify_gpu_ms
    ema_beta=0.2,                       # EMA smoothing for alpha and ms/token
    eps=1e-6,                           # for log stability
    w_weight=None,                      # optional dict[w]->weight (default 1.0)
):
    """
    Proportional-fairness optimizer using myopic log utility:
        maximize sum_i w_i * log(eps + E[A_i(N_i; alpha_i)] / T(N))
    where E[A_i(N; alpha)] = (1 - alpha^(N+1)) / (1 - alpha)  (geometric-prefix model).

    Constraints:
      - min_spec_len <= N_i <= min(max_spec_len, cap_by_time_i)
      - sum_i N_i <= verifier_budget

    Time model (approx):
      - Draft stage time ≈ max_i (ms_per_tok_i * N_i)
      - Verify stage time ≈ verify_base + verify_slope * sum_i N_i
        with verify_slope estimated from last round as verify_gpu_ms / max(1, sum(prev_next_spec_len[w]))
      - Total T ≈ max_draft + verify_slope * sumN
        (constants cancel in ratio so we ignore additive constants by default)
    """
    workers = list(dispatched_workers)
    if not workers:
        return {}

    # ---------- 1) Update EMA alpha + ms/token using observations ----------
    # alpha estimate per round: acc_hat = accepted_len / spec_used (clipped)
    for w in workers:
        accepted_len, spec_used = round_acc[w]
        spec_used = max(1, int(spec_used))
        acc_hat = float(accepted_len) / spec_used
        acc_hat = max(0.0, min(1.0, acc_hat))

        mspt_hat = float(draft_gpu_ms[w]) / spec_used
        mspt_hat = max(1e-6, mspt_hat)

        # EMA updates
        ema_acc[w] = (1.0 - ema_beta) * float(ema_acc[w]) + ema_beta * acc_hat
        # clip alpha away from 0/1 to avoid numerical issues
        ema_acc[w] = min(1.0 - 1e-6, max(1e-6, ema_acc))
        ema_ms_per_tok[w] = (1.0 - ema_beta) * float(ema_ms_per_tok[w]) + ema_beta * mspt_hat
        ema_ms_per_tok[w] = max(1e-6, ema_ms_per_tok[w])

    # ---------- 2) Per-worker time cap based on verify time ----------
    # Want draft stage not dominate: cap_i = floor(time_cap / mspt_i)
    time_cap_ms = float(balance_alpha) * float(verify_gpu_ms)
    cap_by_time = {}
    for w in workers:
        cap = int(time_cap_ms / float(ema_ms_per_tok[w]))
        cap = max(min_spec_len, cap)
        cap_by_time[w] = min(max_spec_len, cap)

    # ---------- 3) Budget feasibility ----------
    B = int(verifier_budget)
    min_total = min_spec_len * len(workers)
    if B <= 0:
        return {w: 0 for w in workers}
    if B < min_total:
        # cannot satisfy minimum for all; fall back to minimum (will exceed B logically, but consistent)
        # alternatively: allocate 0 to some workers; choose your system behavior.
        return {w: min_spec_len for w in workers}

    # ---------- 4) Helper: expected accepted under geometric-prefix model ----------
    def exp_accept(N: int, alpha: float) -> float:
        # E[A] = sum_{j=0..N} alpha^j = (1 - alpha^(N+1)) / (1 - alpha)
        # stable when alpha close to 1: use expm1/log1p trick
        alpha = min(1.0 - 1e-12, max(1e-12, alpha))
        if abs(1.0 - alpha) < 1e-8:
            return float(N + 1)
        # alpha^(N+1)
        a_pow = alpha ** (N + 1)
        return (1.0 - a_pow) / (1.0 - alpha)

    # ---------- 5) Approximate verify-time slope from previous round ----------
    # verify_gpu_ms corresponds to verifying sum(prev_next_spec_len) tokens last round (approx)
    prev_sumN = 0
    for w in workers:
        prev_sumN += int(prev_next_spec_len[w]) if prev_next_spec_len is not None else 0
    prev_sumN = max(1, prev_sumN)
    verify_slope = float(verify_gpu_ms) / prev_sumN  # ms per token verified (approx)

    # ---------- 6) Initialize N with minimums (or warm-start then clamp) ----------
    N = {w: min_spec_len for w in workers}
    # Optional warm-start: start near previous, but must respect caps and budget
    if prev_next_spec_len is not None:
        for w in workers:
            warm = int(prev_next_spec_len[w])
            warm = max(min_spec_len, min(cap_by_time[w], warm))
            N[w] = warm

        # Repair to respect budget B
        def totalN():
            return sum(N.values())

        # If over budget, decrement lowest marginal utility repeatedly
        # We'll just reset to min here for simplicity and stability.
        if totalN() > B:
            N = {w: min_spec_len for w in workers}

    # ---------- 7) Define objective surrogate: sum w_i log(eps + E[A_i]/T) ----------
    # T(N) ≈ max_i (mspt_i * N_i) + verify_slope * sumN
    def compute_T(Ndict) -> float:
        sumN = 0
        max_draft = 0.0
        for ww in workers:
            n = int(Ndict[ww])
            sumN += n
            draft_t = float(ema_ms_per_tok[ww]) * n
            if draft_t > max_draft:
                max_draft = draft_t
        return max_draft + verify_slope * sumN

    def utility(Ndict) -> float:
        T = compute_T(Ndict)
        # Avoid division by 0 (shouldn't happen)
        T = max(1e-9, T)
        val = 0.0
        for ww in workers:
            wgt = 1.0 if w_weight is None else float(w_weight.get(ww, 1.0))
            EA = exp_accept(int(Ndict[ww]), float(ema_acc[ww]))
            rate = EA / T
            val += wgt * math.log(eps + rate)
        return val

    # ---------- 8) Greedy marginal-utility allocation up to budget ----------
    # Start from current N (min or warm) then add tokens until total == B.
    # Each step, try adding +1 token to a feasible worker and pick the best delta utility.
    total = sum(N.values())
    if total < min_total:
        # ensure min for all
        for w in workers:
            N[w] = min_spec_len
        total = sum(N.values())

    # Ensure N respects caps
    for w in workers:
        N[w] = max(min_spec_len, min(cap_by_time[w], int(N[w])))

    total = sum(N.values())
    if total > B:
        # If still over budget, shrink from workers with smallest benefit of having extra tokens.
        # Use greedy removal based on least utility drop.
        while total > B:
            best_w = None
            best_drop = None
            baseU = utility(N)
            for w in workers:
                if N[w] > min_spec_len:
                    N[w] -= 1
                    u2 = utility(N)
                    drop = baseU - u2
                    N[w] += 1
                    if best_drop is None or drop < best_drop:
                        best_drop = drop
                        best_w = w
            if best_w is None:
                break
            N[best_w] -= 1
            total -= 1

    # Now add tokens until budget is filled
    while total < B:
        baseU = utility(N)
        best_w = None
        best_gain = 0.0

        for w in workers:
            if N[w] >= cap_by_time[w]:
                continue
            if N[w] >= max_spec_len:
                continue

            N[w] += 1
            u2 = utility(N)
            gain = u2 - baseU
            N[w] -= 1

            if (best_w is None) or (gain > best_gain):
                best_gain = gain
                best_w = w

        if best_w is None:
            break
        # If no positive gain, still we may want to use remaining budget for throughput;
        # but for pure utility we can stop.
        if best_gain <= 0.0:
            break

        N[best_w] += 1
        total += 1

    return N

def coordinator_loop(args, target_model, target_tokenizer, num_workers, in_qs, out_q):
    """
    Coordinator chạy trên GPU target_gpu: dispatch 3 drafts song song, verify 1 lần batch.
    tqdm theo số câu hoàn thành.
    """
    device0 = torch.device(f"cuda:{args.target_gpu}")

    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token_id = target_tokenizer.eos_token_id

    def make_state(problem_id: int):
        raw_data = format_problem_and_options(args, problem_id)
        messages = [{"role": "user", "content": get_first_user_msg(args, raw_data)}]
        text = target_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = target_tokenizer([text], return_tensors="pt")
        orig_input_ids = enc["input_ids"][0].tolist()
        return {
            "problem_id": problem_id,
            "orig_input_ids": orig_input_ids,
            "current_token_ids": [],
            "done": False,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "num_speculation_rounds": 0,
            "total_output_tokens": 0,
        }

    # chia đều num_questions cho num_workers theo modulo
    dataset_size = args.num_questions
    buckets = [[] for _ in range(num_workers)]
    for pid in range(dataset_size):
        buckets[pid % num_workers].append(pid)
    bucket_ptr = [0] * num_workers

    # mỗi worker giữ 1 active state
    active = [None] * num_workers
    for w in range(num_workers):
        if bucket_ptr[w] < len(buckets[w]):
            active[w] = make_state(buckets[w][bucket_ptr[w]])
            bucket_ptr[w] += 1

    def all_done():
        return all(st is None for st in active)

    # ----------- (3) Optimizer state: EMA & next_spec_len -----------
    # Bạn có thể tune các giá trị default này qua args
    beta = getattr(args, "ema_beta", 0.2)
    ema_acc = [0.7] * num_workers
    ema_ms_per_tok = [20.0] * num_workers  # ước lượng ms/token ban đầu
    next_spec_len = [int(args.spec_len)] * num_workers

    verifier_budget = int(getattr(args, "verifier_budget", num_workers * int(args.spec_len)))
    min_spec_len = int(getattr(args, "min_spec_len", 4))
    max_spec_len = int(getattr(args, "max_spec_len", 16))
    balance_alpha = float(getattr(args, "balance_alpha", 1.0))

    # --- EMA state ---
    ema_beta_metrics = getattr(args, "metrics_ema_beta", 0.2)  # hoặc hardcode 0.2
    ema_accepted_tps = None
    ema_acceptance = None
    ema_straggler = None

    system_stats = []
    pbar = tqdm(total=dataset_size, desc="Math completed", unit="q")
    done_count = 0

    total_wall_s = 0.0
    while not all_done():
        round_t0 = time.perf_counter()

        # 1) dispatch cho các worker còn active
        dispatched_workers = []
        for w in range(num_workers):
            st = active[w]
            if st is None or st["done"]:
                continue

            # quota do optimizer quyết định (round trước)
            quota = int(next_spec_len[w]) if args.optimize_spec_len else int(args.spec_len)
            quota = max(min_spec_len, min(max_spec_len, quota))

            in_qs[w].put({
                "worker_id": w,
                "problem_id": st["problem_id"],
                "orig_input_ids": st["orig_input_ids"],
                "current_token_ids": st["current_token_ids"],
                "drafter_threshold": args.drafter_thresholds[0],
                "spec_len": quota,   # <<< QUAN TRỌNG: gửi quota
            })
            dispatched_workers.append(w)

        if not dispatched_workers:
            break

        # 2) collect draft results (đợi đủ các worker đã dispatch)
        batch_items = []
        draft_gpu_ms = {}
        draft_num_fwd = {}

        for _ in range(len(dispatched_workers)):
            res = out_q.get()  # blocking
            w = res["worker_id"]
            st = active[w]
            if st is None:
                continue

            draft_gpu_ms[w] = float(res["draft_time_ms"])
            draft_num_fwd[w] = int(res["num_forward_passes"])

            batch_items.append({
                "worker_id": w,
                "problem_id": res["problem_id"],
                "orig_input_ids": st["orig_input_ids"],
                "current_token_ids": st["current_token_ids"],
                "draft": res["draft"],
            })

        round_t_after_draft = time.perf_counter()

        # 3) verify batch 1 lần trên target
        seqs = []
        masks = []
        meta = []  # (worker_id, start, end, draft)

        for item in batch_items:
            w = item["worker_id"]
            orig_ids = item["orig_input_ids"]
            cur = item["current_token_ids"]
            draft = item["draft"]

            prompt_len = len(orig_ids)
            prefix_len = len(cur)

            full = orig_ids + cur + draft
            full_t = torch.tensor(full, device=device0, dtype=torch.long)
            attn = torch.ones_like(full_t, dtype=torch.long)

            start = prompt_len + prefix_len - 1
            end = start + len(draft)

            seqs.append(full_t)
            masks.append(attn)
            meta.append((w, start, end, draft))

        batch_input = pad_sequence(seqs, batch_first=True, padding_value=target_tokenizer.pad_token_id)
        batch_mask = pad_sequence(masks, batch_first=True, padding_value=0)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()

        with torch.inference_mode():
            out = target_model(input_ids=batch_input, attention_mask=batch_mask)
            logits = out.logits

        end_evt.record()
        torch.cuda.synchronize()
        verify_gpu_ms = float(start_evt.elapsed_time(end_evt))

        # 4) accept/reject per sample
        round_acc = {}  # worker_id -> (accepted_len, spec_used)
        for b, (w, start, end, draft) in enumerate(meta):
            st = active[w]
            if st is None:
                continue

            verify_logits = logits[b, start:end, :]
            accepted_len = 0
            bonus_token = None

            for i in range(len(draft)):
                pred = torch.argmax(verify_logits[i], dim=-1).item()
                if draft[i] == pred:
                    accepted_len += 1
                else:
                    final_token = pred
                    break
            else:
                final_token = torch.argmax(logits[b, -1, :], dim=-1).item()
                bonus_token = final_token

            tokens_to_append = draft[:accepted_len] + [final_token]
            st["current_token_ids"].extend(tokens_to_append)

            st["accepted_tokens"] += accepted_len
            st["rejected_tokens"] += (len(draft) - accepted_len)
            st["num_speculation_rounds"] += 1
            st["total_output_tokens"] = len(st["current_token_ids"])

            # done conditions
            if (target_tokenizer.eos_token_id in tokens_to_append) or (len(st["current_token_ids"]) >= args.max_new_tokens):
                st["done"] = True

            round_acc[w] = (accepted_len, len(draft))

        round_t1 = time.perf_counter()

        # round_acc: w -> (accepted_len, spec_used)
        accepted_tokens_round = sum(a for (a, s) in round_acc.values())
        drafted_tokens_round = sum(s for (a, s) in round_acc.values())

        acceptance_rate_round = accepted_tokens_round / max(1, drafted_tokens_round)

        round_wall_s = max(1e-9, (round_t1 - round_t0))
        accepted_tok_per_sec_round = accepted_tokens_round / round_wall_s

        # --- EMA updates ---
        if ema_accepted_tps is None:
            ema_accepted_tps = accepted_tok_per_sec_round
        else:
            ema_accepted_tps = ema_accepted_tps + accepted_tok_per_sec_round

        if ema_acceptance is None:
            ema_acceptance = acceptance_rate_round
        else:
            ema_acceptance = ema_acceptance + acceptance_rate_round

        # --- Straggler ratio: max / mean ---
        bs = max(1, len(meta))  # active batch size
        draft_ms_max = max(draft_gpu_ms.values()) if draft_gpu_ms else 0.0
        draft_ms_sum = sum(draft_gpu_ms.values()) if draft_gpu_ms else 0.0
        draft_ms_mean = draft_ms_sum / max(1, bs)

        straggler_ratio = (draft_ms_max / max(1e-9, draft_ms_mean)) if draft_gpu_ms else 0.0
        if ema_straggler is None:
            ema_straggler = straggler_ratio
        else:
            ema_straggler = ema_straggler + straggler_ratio

        # ----------- E) OPTIMIZER: tính quota cho round kế tiếp -----------
        # chỉ tối ưu cho các worker thực sự có trong round_acc
        active_for_opt = [w for w in dispatched_workers if w in round_acc and w in draft_gpu_ms]

        if args.optimize_spec_len:
            proposed = optimizer(
                dispatched_workers=active_for_opt,
                round_acc=round_acc,
                draft_gpu_ms=draft_gpu_ms,
                verify_gpu_ms=verify_gpu_ms,
                ema_acc=ema_acc,
                ema_ms_per_tok=ema_ms_per_tok,
                prev_next_spec_len=next_spec_len,
                verifier_budget=verifier_budget,
                min_spec_len=min_spec_len,
                max_spec_len=max_spec_len,
                balance_alpha=balance_alpha,
                ema_beta=beta,
            )
            for w, v in proposed.items():
                next_spec_len[w] = int(v)
        else:
            # tắt optimize -> luôn reset về spec_len cố định
            for w in dispatched_workers:
                next_spec_len[w] = int(args.spec_len)

        # 5) refill finished workers + tqdm update (CHỈ update ở đây)
        for w in range(num_workers):
            st = active[w]
            if st is None:
                continue
            if st["done"]:
                done_count += 1
                pbar.update(1)

                if bucket_ptr[w] < len(buckets[w]):
                    active[w] = make_state(buckets[w][bucket_ptr[w]])
                    bucket_ptr[w] += 1
                    next_spec_len[w] = int(args.spec_len)
                else:
                    active[w] = None

        # 6) timing stats + tqdm postfix
        round_idx = len(system_stats)
        system_stats.append({
            "round_wall_ms": (round_t1 - round_t0) * 1000.0,
            "draft_wait_wall_ms": (round_t_after_draft - round_t0) * 1000.0,
            "draft_gpu_ms_max": max(draft_gpu_ms.values()) if draft_gpu_ms else 0.0,
            "draft_gpu_ms_sum": sum(draft_gpu_ms.values()) if draft_gpu_ms else 0.0,
            "verify_gpu_ms": verify_gpu_ms,
            "draft_num_fwd_sum": int(sum(draft_num_fwd.values())) if draft_num_fwd else 0,
            "active_batch_size": int(len(meta)),
            "done": int(done_count),
            "next_spec_len": {int(w): int(next_spec_len[w]) for w in dispatched_workers},
            "ema_acc": {int(w): float(round(ema_acc[w], 4)) for w in dispatched_workers},
            "ema_mspt": {int(w): float(round(ema_ms_per_tok[w], 4)) for w in dispatched_workers},
            "acceptance_rate_round": acceptance_rate_round,
            "accepted_tok_per_sec_round": accepted_tok_per_sec_round,
            "ema_acceptance_rate": ema_acceptance/max(round_idx,1),
            "ema_accepted_tok_per_sec": ema_accepted_tps/max(round_idx,1),
            "ema_straggler_ratio": ema_straggler/max(round_idx,1),
            "draft_straggler_ratio": straggler_ratio/max(round_idx,1),
        })

        last = system_stats[-1]
        pbar.set_postfix({
            "round_ms": f"{last['round_wall_ms']:.1f}",
            "draft_ms": f"{last['draft_gpu_ms_max']:.1f}",
            "verify_ms": f"{last['verify_gpu_ms']:.1f}",
            "bs": last["active_batch_size"],
            "B": verifier_budget,
        })

        round_idx = len(system_stats) - 1
        bs = last["active_batch_size"]
        if (round_idx > 0) and (bs == 3):
            safe_wandb_log({
                "round/idx": len(system_stats),                     # round index (1-based nếu bạn muốn +1)
                "round/wall_ms": last["round_wall_ms"],             # total round wall time (ms)
                "round/acceptance_rate": acceptance_rate_round,     # acceptance rate round
                "round/accepted_tok_per_sec": accepted_tok_per_sec_round,  # accepted token per second
                "round/accepted_tokens": accepted_tokens_round,
                "round/drafted_tokens": drafted_tokens_round,
                "round/active_batch_size": last["active_batch_size"],
                "progress/done": last["done"],
                "ema/accepted_tok_per_sec": ema_accepted_tps/round_idx,
                "ema/acceptance_rate": ema_acceptance/round_idx,
                "ema/traggler_ratio": ema_straggler/round_idx,
                "straggler/ratio": straggler_ratio,
                "round/bs": bs,
            })
        total_wall_s += round_wall_s

    safe_wandb_log({"total/wall_s": total_wall_s}, commit=False)

    pbar.close()
    return system_stats

# %%
def safe_wandb_log(payload, commit=True):
    if wandb.run is not None:
        wandb.log(payload, commit=commit)


def init_wandb(args):
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        logging.info("wandb disabled")
        return None

    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ.pop("WANDB_MODE", None)

    try:
        run = wandb.init(
            project="specdiff_aoi",
            name=f"{args.dataset_name}_multi_gpu_opt{int(args.optimize_spec_len)}",
            config={
                "dataset": args.dataset_name,
                "num_questions": args.num_questions,
                "spec_len": args.spec_len,
                "target_model": args.target_model_name,
                "dllm_dir": args.dllm_dir,
                "verifier_budget": getattr(args, "verifier_budget", None),
                "optimize_spec_len": getattr(args, "optimize_spec_len", False),
                "num_workers": args.num_drafters,
                "wandb_mode": args.wandb_mode,
            },
        )
        return run
    except Exception as exc:
        logging.warning("wandb init failed, fallback to disabled mode: %s", exc)
        os.environ["WANDB_DISABLED"] = "true"
        return None


parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gsm8k", "gpqa", "humaneval"], default="math",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="./outputs", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--dllm_dir", type=str, default=None,
                    help="Dir to the dLLM weights and (modified) modeling.py")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max new tokens from the target model")
parser.add_argument("--block_size", type=int, default=32,
                    help="Block size in Fast-dLLM")
parser.add_argument("--spec_len", type=int, default=10,
                    help="Frequency of verification steps (in number of tokens)")
parser.add_argument("--drafter_thresholds", type=float, nargs="+",  # one or more float thresholds
                    default=[0.05],
                    help="Threshold for confidence-adaptive decoding of the dLLM drafter model (e.g., --drafter_thresholds 0.1 0.5 0.9 runs a sweep of the three)")
parser.add_argument("--log_level",
                    type=str,
                    default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the logging level")
parser.add_argument("--sweep_lowconf_threshold", type=float, nargs="+",
                    default=[0.4],
                    help="τ in FailFast Alg. 1")
parser.add_argument("--sweep_max_spec_len", type=int, nargs="+",
                    default=[60],
                    help="N_max in FailFast Alg. 1")
parser.add_argument("--sweep_incr_len", type=int, nargs="+",
                    default=[10],
                    help="N in FailFast Alg. 1")
parser.add_argument('--run_ar', action='store_true', help='Run the AR drafter to compare speedups')
parser.add_argument('--run_dllm_sf', action='store_true', help='Run the dLLM drafter with static frequency (in param sweep for baselines)')
parser.add_argument('--baseline_sweep', action='store_true', help='Running a baseline sweep, don\'t run dynamic frequency')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output pickles and figures')
parser.add_argument('--reuse_drafts', action='store_true', help='Reuses drafted tokens from previous rounds if possible -- see Appendix E.')
parser.add_argument('--disable_reusing_drafter_kvs', action='store_true', help='Disables reusing drafter KV cache across verification rounds')
parser.add_argument('--read_pickle', action='store_true', help='Use acceptance decisions from a cached pickle file rather than rerunning')
parser.add_argument("--multi_gpu", action="store_true", help="Run 1 target (GPU0) + N drafters (GPU1..)")
parser.add_argument("--num_drafters", type=int, default=3, help="Number of drafter workers (default 3)")
parser.add_argument("--target_gpu", type=int, default=0, help="GPU id for target model (default 0)")
parser.add_argument("--drafter_gpus", type=int, nargs="+", default=[1,2,3], help="GPU ids for drafter workers (default 1 2 3)")
parser.add_argument("--timing_out", type=str, default="./outputs/system_timing.jsonl", help="Where to write timing jsonl")
parser.add_argument("--verifier_budget", type=int, default=24,
                    help="Max total draft tokens verified per target call (sum over batch)")
parser.add_argument("--max_spec_len", type=int, default=16)
parser.add_argument("--min_spec_len", type=int, default=4)
parser.add_argument("--balance_alpha", type=float, default=1.0,
                    help="time_cap = alpha * verify_gpu_ms (controls waiting vs verify balance)")
parser.add_argument("--optimize_spec_len", action="store_true",
                    help="Enable coordinator optimizer to adapt next spec_len per drafter")
parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="offline",
                    help="wandb mode for deployment envs: online/offline/disabled")

args, _ = parser.parse_known_args()
args.target_model_name_clean = args.target_model_name.split("/", 1)[1]

######custom fields for easier debugging######
# args.log_level = "DEBUG"
# args.disable_reusing_drafter_kvs = True
# args.dataset_name = "gpqa"
# args.overwrite = True
# args.max_new_tokens = 1024
# args.run_ar = True
# args.baseline_sweep = True
# args.spec_len = 8
# args.run_dllm_sf = True
# args.read_pickle = True  # XXX: read trajectory from pickle as well in future debugging
# args.target_model_name = "Qwen/Qwen2.5-7B-Instruct"  # for easier debugging
# args.sweep_lowconf_threshold = [0.4]
# args.sweep_max_spec_len = [50]
# args.sweep_incr_len = [10]
######custom fields for easier debugging######


logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
    # datefmt="%m%d",
)

if args.num_drafters < 1:
    parser.error("--num_drafters must be >= 1")

if len(args.drafter_gpus) < args.num_drafters:
    parser.error("--drafter_gpus must provide at least --num_drafters GPU ids")

if (args.run_dllm_sf or (not args.baseline_sweep)) and not args.dllm_dir:
    parser.error("--dllm_dir is required when running dLLM drafter")

run = init_wandb(args)

construct_drafter_configs(args)  # populates args.drafter_configs
populate_dataset(args)  # populates args.dataset

args.latency = {  # all in ms
    "vLLM_A6000": {
        "draft_fwd_pass": 6.1,
        "target_tpt": {
            "Qwen2.5-7B-Instruct": 13.5,
            "Qwen2.5-14B-Instruct": 24.7,
            "Qwen2.5-32B-Instruct": 52.6,
        },
    },
    # "vLLM_H100": {  # profiling was inaccurate
    #     "draft_fwd_pass": 2.9,  # eager mode: 9.25?
    #     "target_tpt": {  # eager mode on
    #         "Qwen2.5-14B-Instruct": 14.3,  # w/o eager mode: 14.3. 8.45??
    #         "Qwen2.5-32B-Instruct": 18.6,
    #         "Qwen2.5-72B-Instruct": 32.2,
    #     },
    # },
}

target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
args.target_tokenizer = target_tokenizer


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # load target on cuda:0
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name,
        torch_dtype="auto",
        device_map={"": args.target_gpu},
    ).eval()

    num_workers = args.num_drafters
    drafter_gpus = args.drafter_gpus[:num_workers]

    in_qs = [mp.Queue() for _ in range(num_workers)]
    out_q = mp.Queue()

    procs = []
    for w in range(num_workers):
        p = mp.Process(target=drafter_worker, args=(w, drafter_gpus[w], args, in_qs[w], out_q))
        p.start()
        procs.append(p)

    stats = coordinator_loop(args, target_model, target_tokenizer, num_workers, in_qs, out_q)

    # shutdown workers
    for q in in_qs:
        q.put(None)
    for p in procs:
        p.join()

    os.makedirs(os.path.dirname(args.timing_out), exist_ok=True)
    with open(args.timing_out, "w") as f:
        for row in stats:
            f.write(json.dumps(row) + "\n")

    print(f"[OK] Wrote timing to {args.timing_out}")
    if wandb.run is not None:
        wandb.finish()
    sys.exit(0)
