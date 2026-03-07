# target on GPU0
target_model = AutoModelForCausalLM.from_pretrained(..., device_map={"":0})
target_model.eval()

# create 3 queues in/out
in_qs = [mp.Queue() for _ in range(3)]
out_q = mp.Queue()

# spawn 3 workers on GPU1/2/3
procs = []
for k, gpu in enumerate([1,2,3]):
    p = mp.Process(target=drafter_worker, args=(k, gpu, args, in_qs[k], out_q))
    p.start()
    procs.append(p)

# build 3 sample states (3 different problem_id / data)
states = [...]  # each state holds orig_input_ids, current_token_ids, prev_prefill_output, done...

system_stats = []

while not all(st["done"] for st in states):
    round_t0 = time.perf_counter()

    # 1) dispatch draft tasks
    active_indices = [i for i, st in enumerate(states) if not st["done"]]
    # nếu <3 active, bạn có thể chỉ dispatch cho số đó và verify batch nhỏ hơn
    for worker_slot, i in enumerate(active_indices[:3]):
        st = states[i]
        in_qs[worker_slot].put({
            "sample_id": i,
            "orig_input_ids": st["orig_input_ids"],
            "current_token_ids": st["current_token_ids"],
            "prev_prefill_output": st.get("prev_prefill_output", None),
            "drafter_threshold": st["drafter_threshold"],
        })

    # 2) collect drafts (đợi đủ)
    batch_items = []
    draft_times = []
    total_fwd = 0
    for _ in range(len(active_indices[:3])):
        res = out_q.get()
        i = res["sample_id"]
        states[i]["prev_prefill_output"] = res.get("prefill_output", None)
        total_fwd += res["num_forward_passes"]
        draft_times.append(res["draft_time_ms"])
        batch_items.append({
            "orig_input_ids": states[i]["orig_input_ids"],
            "current_token_ids": states[i]["current_token_ids"],
            "draft": res["draft"],
        })

    round_t_after_draft = time.perf_counter()

    # 3) verify once on GPU0 (batched)
    updates, verify_time_ms = verify_batch_target(target_model, target_tokenizer, batch_items, num_target_tokens=args.max_new_tokens)

    # 4) apply updates back to states
    for item, upd in zip(active_indices[:3], updates):
        st = states[item]
        st["current_token_ids"].extend(upd["tokens_to_append"])
        # done condition
        if (target_tokenizer.eos_token_id in upd["tokens_to_append"]) or (len(st["current_token_ids"]) >= args.max_new_tokens):
            st["done"] = True

    round_t1 = time.perf_counter()

    system_stats.append({
        "round_wall_ms": (round_t1 - round_t0)*1000,
        "draft_wait_wall_ms": (round_t_after_draft - round_t0)*1000,
        "draft_gpu_ms_max": max(draft_times) if draft_times else 0.0,  # parallel => bottleneck is max
        "draft_gpu_ms_sum": sum(draft_times),
        "verify_gpu_ms": verify_time_ms,
        "num_forward_passes_draft": total_fwd,
    })

# shutdown workers
for q in in_qs: q.put(None)
for p in procs: p.join()
