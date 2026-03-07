from torch.nn.utils.rnn import pad_sequence

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
