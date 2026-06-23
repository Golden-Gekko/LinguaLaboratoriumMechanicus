import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


def top_k_filtering(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    threshold = torch.topk(
        logits, min(top_k, logits.size(-1)), dim=-1
    ).values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float('-inf'))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    device: str = 'cpu',
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = input_ids.size(1)
    generated = input_ids.clone()
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if generated.size(1) > model.max_context_length:
            generated = generated[:, -model.max_context_length:]

        logits = model(generated)
        logits_last = logits[0, -1, :] / temperature
        logits_last = top_k_filtering(logits_last.unsqueeze(0), top_k).squeeze(0)

        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=-1)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

        if next_id.item() == eos_token_id:
            break

    model.train()
    return tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
