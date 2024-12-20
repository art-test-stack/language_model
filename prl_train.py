from michelgpt.train.trainer_fsdp import FSDPTrainer

if __name__ == "__main__":
    # To clear cache: rm -rf ~/.cache/huggingface/datasets/
    trainer = FSDPTrainer()
    # trainer.train()