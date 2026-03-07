import torch, time
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

def drafter_worker(
    rank, gpu_id, args, in_q: mp.Queue, out_q: mp.Queue
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load drafter model onto its GPU
    dllm = AutoModelForCausalLM.from_pretrained(
        args.dllm_dir,
        torch_dtype="auto",
        device_map={"": gpu_id},   # quan trọng: ép lên đúng GPU
        trust_remote_code=True,
    )
    dllm.eval()

    while True:
        msg = in_q.get()
        if msg is None:
            break

        sample_id = msg["sample_id"]
        orig_input_ids = msg["orig_input_ids"]        # list[int]
        current_token_ids = msg["current_token_ids"]  # list[int]
        prev_prefill = msg.get("prev_prefill_output", None)

        # Build orig_model_inputs on this GPU
        orig_model_inputs = {
            "input_ids": torch.tensor([orig_input_ids], device=device, dtype=torch.long),
            "attention_mask": torch.ones((1, len(orig_input_ids)), device=device, dtype=torch.long),
        }

        # --- timing draft (GPU time) ---
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()

        # Call your existing helper
        draft, prefill_output, num_fwd, lat_list = get_next_n_tokens_dllm(
            dllm, args, orig_model_inputs, current_token_ids,
            spec_len=args.spec_len,
            output_seqlen=3*args.block_size,
            small_block_size=8,
            threshold=msg["drafter_threshold"],
            is_drafter=True,
            prev_prefill_output=prev_prefill,
        )

        end_evt.record()
        torch.cuda.synchronize()
        draft_time_ms = start_evt.elapsed_time(end_evt)

        out_q.put({
            "sample_id": sample_id,
            "draft": draft,  # list[int]
            "prefill_output": prefill_output,
            "num_forward_passes": num_fwd,
            "draft_time_ms": draft_time_ms,
        })
