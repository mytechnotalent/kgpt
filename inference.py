import torch
import warnings
from model import GPT, enc, block_size

warnings.filterwarnings("ignore")

# select the best available compute device prioritizing cuda then mps then cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# the max_new_tokens parameter controls the maximum number of tokens the model
# will generate in a single response preventing excessively long outputs
max_new_tokens = 128
# the temperature parameter controls the randomness of token sampling where
# lower values make the model more deterministic and higher values more creative
temperature = 0.3
# the top_k parameter limits the sampling pool to the k most probable next tokens
# filtering out low-probability tokens that could produce incoherent text
top_k = 20
# the repetition_penalty parameter discourages the model from repeating tokens
# by dividing the logits of previously generated tokens by this value
repetition_penalty = 1.3


def _load_model(checkpoint_path):
    """
    Loads the fine-tuned model weights from a checkpoint file on disk and moves
    the model to the target compute device in evaluation mode.

    Args:
        checkpoint_path (str): The file path to the fine-tuned model checkpoint
            saved by finetune.py containing the state dictionary.

    Returns:
        GPT: The model instance with loaded weights on the target device in
            evaluation mode ready for inference.
    """
    model = GPT()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


def _build_prompt_tokens(conversation_history):
    """
    Encodes the full conversation history into a token ID list using the chat
    format delimiters that the model learned during fine-tuning.

    Args:
        conversation_history (list): A list of (role, text) tuples where role is
            either 'user' or 'assistant' representing the conversation so far.

    Returns:
        list: A list of integer token IDs representing the formatted conversation
            history ready to be fed to the model for generation.
    """
    prompt = ""
    for role, text in conversation_history:
        prompt += f"<|{role}|>\n{text}\n"
    prompt += "<|assistant|>\n"
    return enc.encode(prompt)


def _apply_repetition_penalty(logits, generated_ids):
    """
    Applies a repetition penalty to the logits by dividing the scores of tokens
    that have already been generated making them less likely to be repeated.

    Args:
        logits (torch.Tensor): The raw logits tensor of shape (vocab_size,) from
            the model's last position output.
        generated_ids (list): A list of token IDs that have already been generated
            in the current response.

    Returns:
        torch.Tensor: The modified logits tensor with repetition penalty applied
            to previously generated token positions.
    """
    if not generated_ids:
        return logits
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= repetition_penalty
        else:
            logits[token_id] *= repetition_penalty
    return logits


def _sample_next_token(logits):
    """
    Samples the next token from the logits distribution using temperature scaling
    and top-k filtering for controlled randomness in generation.

    Args:
        logits (torch.Tensor): The raw logits tensor of shape (vocab_size,) from
            the model's output at the last sequence position.

    Returns:
        int: The sampled token ID as a Python integer selected from the filtered
            and temperature-scaled probability distribution.
    """
    logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(logits, top_k)
    probs = torch.softmax(top_k_values, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    return top_k_indices[sampled_index].item()


def _is_end_token(token_id):
    """
    Checks whether a generated token ID corresponds to an end-of-conversation or
    end-of-turn delimiter that should terminate response generation.

    Args:
        token_id (int): The token ID to check against known end delimiter strings.

    Returns:
        bool: True if the token represents an end-of-conversation marker or a user
            turn start marker indicating the model is trying to generate a new turn.
    """
    token_text = enc.decode([token_id])
    return "<|end|>" in token_text or "<|user|>" in token_text


def _generate_response(model, prompt_tokens):
    """
    Generates a complete response token sequence from the model given a prompt
    using autoregressive decoding with temperature, top-k, and repetition penalty.

    Args:
        model (GPT): The fine-tuned model instance in eval mode.
        prompt_tokens (list): A list of integer token IDs representing the
            formatted conversation prompt.

    Returns:
        str: The decoded text string of the model's generated response with
            leading and trailing whitespace stripped.
    """
    idx = torch.tensor([prompt_tokens[-block_size:]], dtype=torch.long, device=device)
    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            logits = _apply_repetition_penalty(logits[0], generated_ids)
            next_token = _sample_next_token(logits)
            if _is_end_token(next_token):
                break
            generated_ids.append(next_token)
            idx = torch.cat([idx, torch.tensor([[next_token]], device=device)], dim=1)
    return enc.decode(generated_ids).strip()


def _truncate_to_one_sentence(text):
    """
    Truncates text to the first complete sentence by splitting on sentence
    ending punctuation followed by a space or end of string, avoiding cuts
    inside decimal numbers or abbreviations.

    Args:
        text (str): The full response text that may contain multiple sentences.

    Returns:
        str: The first sentence from the input text including its ending
            punctuation mark or the full text if no boundary is found.
    """
    for i, ch in enumerate(text):
        if ch in ".!?" and i > 0:
            after_end = i == len(text) - 1 or text[i + 1] == " "
            before_is_digit = text[i - 1].isdigit()
            after_is_digit = i < len(text) - 1 and text[i + 1].isdigit()
            if after_end and not (before_is_digit and after_is_digit):
                return text[: i + 1].strip()
    return text.strip()


def _extract_clean_response(raw_response):
    """
    Cleans up the raw generated response by removing any residual delimiter tokens
    or formatting artifacts that may have leaked into the output text.

    Args:
        raw_response (str): The raw decoded text from the model's generation output
            that may contain unwanted delimiter strings.

    Returns:
        str: The cleaned response text with all chat format delimiters removed and
            extra whitespace stripped for display to the user.
    """
    for tag in ["<|user|>", "<|assistant|>", "<|end|>"]:
        raw_response = raw_response.split(tag)[0]
    return _truncate_to_one_sentence(raw_response.strip())


def _print_welcome():
    """
    Prints the welcome banner and usage instructions to the console at the start
    of an interactive chat session with the KGPT chatbot.

    Returns:
        None: Prints the welcome message and instructions as a side effect.
    """
    print("\n" + "=" * 60)
    print("  KGPT Chatbot — Powered by a GPT-2-class model")
    print("  Type your message and press Enter to chat.")
    print("  Commands: 'quit' to exit, 'clear' to reset history")
    print("=" * 60 + "\n")


def _run_chat_loop(model):
    """
    Runs the main interactive chat loop that reads user input, generates model
    responses using the full conversation history, and handles special commands.

    Args:
        model (GPT): The fine-tuned model instance in eval mode ready to generate
            conversational responses.

    Returns:
        None: Runs an interactive loop until the user types a quit command.
    """
    conversation_history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "q", "exit"}:
            print("KGPT: Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation_history.clear()
            print("KGPT: Conversation history cleared.\n")
            continue
        conversation_history.append(("user", user_input))
        prompt_tokens = _build_prompt_tokens(conversation_history)
        raw_response = _generate_response(model, prompt_tokens)
        response = _extract_clean_response(raw_response)
        if not response:
            response = "I'm not sure how to respond to that. Could you rephrase?"
        conversation_history.append(("assistant", response))
        print(f"KGPT: {response}\n")


# load the fine-tuned model checkpoint from disk into evaluation mode
model = _load_model("kgpt_finetuned.pt")
print(
    f"Model loaded on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)"
)

# print the welcome banner and start the interactive chat loop
_print_welcome()
_run_chat_loop(model)
