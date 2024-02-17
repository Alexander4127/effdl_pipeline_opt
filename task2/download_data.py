from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', ignore_verifications=True)
    dataset.save_to_disk("data.hf")
