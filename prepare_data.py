import os
import numpy as np
import tiktoken
from datasets import load_dataset


def _get_tokenizer():
    """
    Creates and returns the GPT-2 BPE tokenizer from tiktoken.

    Returns:
        tiktoken.Encoding: The GPT-2 tokenizer encoding object.
    """
    return tiktoken.get_encoding("gpt2")


def _tokenize_example(example, enc):
    """
    Tokenizes a single dataset example text field and appends an end-of-text token.

    Args:
        example (dict): A dataset row containing a 'text' field.
        enc (tiktoken.Encoding): The tokenizer encoding object.

    Returns:
        dict: A dictionary with key 'ids' containing the token list and 'len'
            containing the token count.
    """
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


def _download_dataset():
    """
    Downloads the OpenWebText dataset from Hugging Face and returns it.

    Returns:
        datasets.DatasetDict: The raw OpenWebText dataset split.
    """
    return load_dataset("openwebtext", trust_remote_code=True)


def _split_dataset(dataset):
    """
    Splits the OpenWebText train set into train and validation subsets.

    Args:
        dataset (datasets.DatasetDict): The raw OpenWebText dataset.

    Returns:
        datasets.DatasetDict: A dataset dict with 'train' and 'val' splits where
            the validation set is 0.5 percent of the original training data.
    """
    split_dataset = dataset["train"].train_test_split(
        test_size=0.005, seed=2357, shuffle=False
    )
    split_dataset["val"] = split_dataset.pop("test")
    return split_dataset


def _tokenize_split(split_dataset, enc):
    """
    Applies tokenization across all examples in both train and val splits.

    Args:
        split_dataset (datasets.DatasetDict): Dataset with 'train' and 'val' splits.
        enc (tiktoken.Encoding): The tokenizer encoding object.

    Returns:
        datasets.DatasetDict: The tokenized dataset with 'ids' and 'len' columns
            and the original 'text' column removed.
    """
    tokenized = split_dataset.map(
        lambda example: _tokenize_example(example, enc),
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=os.cpu_count(),
    )
    return tokenized


def _write_binary(tokenized, split, output_dir):
    """
    Writes tokenized data for a given split to a memory-mapped numpy binary file.

    Args:
        tokenized (datasets.DatasetDict): The tokenized dataset.
        split (str): The split name to write ('train' or 'val').
        output_dir (str): The directory path where the binary file will be saved.

    Returns:
        None: Writes the file to disk as a side effect.
    """
    dset = tokenized[split]
    total_len = np.sum(dset["len"], dtype=np.uint64)
    filename = os.path.join(output_dir, f"{split}.bin")
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(total_len,))
    idx = 0
    for example in dset:
        arr[idx : idx + example["len"]] = example["ids"]
        idx += example["len"]
    arr.flush()
    print(f"{split}.bin: {total_len:,} tokens written")


def prepare_data():
    """
    Downloads OpenWebText, tokenizes it with GPT-2 BPE, and saves train.bin
    and val.bin to the data directory for efficient training data loading.

    Returns:
        None: Writes train.bin and val.bin to the 'data' directory as a side effect.
    """
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    enc = _get_tokenizer()
    dataset = _download_dataset()
    split_dataset = _split_dataset(dataset)
    tokenized = _tokenize_split(split_dataset, enc)
    _write_binary(tokenized, "train", output_dir)
    _write_binary(tokenized, "val", output_dir)


if __name__ == "__main__":
    prepare_data()
