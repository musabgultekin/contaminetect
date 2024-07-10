import json
import os

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import fire
from omegaconf import OmegaConf
import jmespath


def find_contaminated_examples(conf, training_examples, training_embeddings, evaluation_examples, evaluation_embeddings):
    """
    Find contaminated examples and return them sorted by similarity.
    """
    contaminated_examples = []
    
    # Calculate similarity and identify contamination
    for i, evaluation_embedding in enumerate(evaluation_embeddings):
        similarity = np.dot(evaluation_embedding, training_embeddings.T)
        max_idx = similarity.argmax().item()
        max_similarity = similarity.max().item()
        
        if max_similarity > conf.embedding.similarity_threshold:
            contaminated_example = {
                "eval": evaluation_examples[i],
                "train": training_examples[max_idx],
                "similarity": max_similarity
            }
            contaminated_examples.append(contaminated_example)
    
    # Sort contaminated examples by similarity in descending order
    contaminated_examples = sorted(contaminated_examples, key=lambda x: x["similarity"], reverse=True)
    return contaminated_examples


def compute_embeddings(model, examples, prompt, batch_size, show_progress_bar):
    """
    Compute embeddings for a list of examples.
    Load from cache if its available.
    """
    
    cache_path = f"cache/{prompt.replace(' ', '_')}.npy"
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
    else:
        embeddings = model.encode(examples, prompt=prompt, batch_size=batch_size, show_progress_bar=show_progress_bar)
        np.save(cache_path, embeddings)

    return embeddings


def process_datasets(conf, model, training_dataset_conf, evaluation_dataset_conf):
    """
    Load datasets and compute contamination between them.
    """
    # Load and process training dataset
    training_dataset = load_dataset(**training_dataset_conf.args)#.shuffle().select(range(1000))
    training_examples = [jmespath.search(training_dataset_conf.prompt_jmespath, example) for example in training_dataset]
    training_embeddings = compute_embeddings(model, training_examples, conf.embedding.prompt, conf.embedding.batch_size, conf.debug)

    # Load and process evaluation dataset
    evaluation_dataset = load_dataset(**evaluation_dataset_conf.args)
    evaluation_examples = [jmespath.search(evaluation_dataset_conf.prompt_jmespath, example) for example in evaluation_dataset]
    evaluation_embeddings = model.encode(evaluation_examples, prompt=conf.embedding.prompt, batch_size=conf.embedding.batch_size, show_progress_bar=conf.debug)

    # Identify contaminated examples
    contaminated_examples = find_contaminated_examples(conf, training_examples, training_embeddings, evaluation_examples, evaluation_embeddings)
    
    # Compute contamination percentage
    contamination_percentage = (len(contaminated_examples) / len(evaluation_examples)) * 100
    
    # Debugging output
    if conf.debug:
        for example in contaminated_examples:
            print(json.dumps(example, indent=4))
        print(f"Contamination: {len(contaminated_examples)}/{len(evaluation_examples)} ({contamination_percentage:.2f}%)")
    
    return contaminated_examples, contamination_percentage


def save_results(output_dir, eval_dataset_name, train_dataset_name, contamination_percentage, contaminated_examples):
    """
    Save contamination results to files.
    """
    # Append contamination percentage to accuracy.csv
    with open(os.path.join(output_dir, "accuracy.csv"), "a") as f:
        f.write(f"{eval_dataset_name},{train_dataset_name},{contamination_percentage}\n")
    
    # Save detailed contaminated examples to a JSONL file
    filepath = f"{eval_dataset_name.replace('/', '_')}-{train_dataset_name.replace('/', '_')}.jsonl"
    with open(os.path.join(output_dir, filepath), "w") as f:
        for example in contaminated_examples:
            f.write(json.dumps(example) + "\n")


def main():
    """
    Main function to run the contamination detection process.
    """
    # Load configuration
    conf = OmegaConf.load("config.yaml")
    
    # Initialize the sentence transformer model
    model = SentenceTransformer(conf.embedding.model)
    
    # Ensure the output directory exists
    os.makedirs(conf.output_dir, exist_ok=True)

    # Initialize the accuracy.csv file
    with open(os.path.join(conf.output_dir, "accuracy.csv"), "w") as f:
        f.write("eval_dataset,train_dataset,contamination_percentage\n")

    # Process each combination of training and evaluation datasets
    for training_dataset_conf in conf.training_datasets:
        for evaluation_dataset_conf in conf.evaluation_datasets:
            contaminated_examples, contamination_percentage = process_datasets(conf, model, training_dataset_conf, evaluation_dataset_conf)
            save_results(conf.output_dir, evaluation_dataset_conf.args['path'], training_dataset_conf.args['path'], contamination_percentage, contaminated_examples)


if __name__ == '__main__':
    fire.Fire(main)