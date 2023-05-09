# removes all the generated stuff when we are testing
rm *.pkl
rm *.pt
rm progress-*.txt
rm total_reward-*.txt
rm actions_chosen-*.txt
rm loss-*.txt

# wandb sync --clean   