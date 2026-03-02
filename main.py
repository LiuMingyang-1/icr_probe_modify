import torch
from src.icr_score import ICRScore
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_cached(
    model_name: str,
    texts,
    device: str = None,
    dtype: torch.dtype = None,
):
    """
    Forward pass and return hidden_states and attentions.

    Args:
        model_name (str): HuggingFace model name or local path
        texts (str or list[str]): input text(s)
        device (str, optional): 'cuda' or 'cpu'
        dtype (torch.dtype, optional): torch.float16 / torch.float32

    Returns:
        hidden_states: tuple(n_layers+1) of (batch, seq_len, hidden_dim)
        attentions: tuple(n_layers) of (batch, n_heads, seq_len, seq_len)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    # Ensure texts is list
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    return hidden_states, attentions


def compute_icr_scores(hidden_states, attentions):
    icr_calculator = ICRScore(
        hidden_states=hidden_states,
        attentions=attentions,
        skew_threshold=0,
        entropy_threshold=1e5,
        core_positions={
            'user_prompt_start': start_position,  
            'user_prompt_end': end_position,  
            'response_start': response_start_position,  
        },
        icr_device='cuda'
    )
    icr_scores, top_p_mean = icr_calculator.compute_icr(
        top_k=20,
        top_p=0.1, 
        pooling='mean',
        attention_uniform=False,
        hidden_uniform=False,
        use_induction_head=True
    )
    return icr_scores, top_p_mean

if __name__ == "__main__":
    model_name = "/data/sjx/models/Meta-Llama-3-8B-Instruct"
    texts = "Hello, how are you?"
    hidden_states, attentions = get_cached(model_name, texts)
    icr_scores, top_p_mean = compute_icr_scores(hidden_states, attentions)
    print(icr_scores)
    print(top_p_mean)
