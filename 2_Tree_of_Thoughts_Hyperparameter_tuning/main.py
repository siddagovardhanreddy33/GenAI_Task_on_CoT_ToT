# Main.py

from train_model import train_and_evaluate
from analyze_results import analyze_results


def main():

    print("\nTraining Model...")
    train_and_evaluate()

    print("\nRunning Tree-of-Thought Analysis...")
    analyze_results()


if __name__ == "__main__":
    main()