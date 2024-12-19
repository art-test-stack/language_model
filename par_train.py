from michelgpt.train.trainer_parallel import ParallelTrainer

if __name__ == "__main__":
    # To clear cache: rm -rf ~/.cache/huggingface/datasets/
    trainer = ParallelTrainer()
    # trainer.train()