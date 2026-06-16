# analyze_results



import pandas as pd


def analyze_results():

    df = pd.read_csv(
        "outputs/hyperparameter_results.csv"
    )

    print("\n" + "=" * 60)
    print("TREE OF THOUGHT ANALYSIS")
    print("=" * 60)

    low = df[df["max_depth"] == 3]
    medium = df[df["max_depth"] == 5]
    high = df[df["max_depth"] == 10]

    print("\nROOT:")
    print("Find Best Hyperparameter Configuration")

    print("\n├── Branch A (Low Complexity)")
    print("│   max_depth = 3")

    best_low = low.sort_values(
        "CV_Accuracy",
        ascending=False
    ).iloc[0]

    print(
        f"│   Best Accuracy = {best_low['CV_Accuracy']}"
    )
    print(
        f"│   Gap = {best_low['Gap']}"
    )

    if best_low["Gap"] < 0.03:
        print("│   Good Generalization")
    else:
        print("│   Mild Overfitting")

    print("│   Possible Underfitting")

    print("\n├── Branch B (Medium Complexity)")
    print("│   max_depth = 5")

    best_medium = medium.sort_values(
        "CV_Accuracy",
        ascending=False
    ).iloc[0]

    print(
        f"│   Best Accuracy = {best_medium['CV_Accuracy']}"
    )
    print(
        f"│   Gap = {best_medium['Gap']}"
    )

    print("│   Strong Balance")
    print("│   Good Bias-Variance Tradeoff")

    print("\n└── Branch C (High Complexity)")
    print("    max_depth = 10")

    best_high = high.sort_values(
        "CV_Accuracy",
        ascending=False
    ).iloc[0]

    print(
        f"    Best Accuracy = {best_high['CV_Accuracy']}"
    )
    print(
        f"    Gap = {best_high['Gap']}"
    )

    if best_high["Gap"] > 0.03:
        print("    Overfitting Risk")
    else:
        print("    Acceptable")

    best = df.sort_values(
        ["CV_Accuracy", "Gap"],
        ascending=[False, True]
    ).iloc[0]

    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)

    print(f"Best Configuration: {best['Config']}")
    print(f"n_estimators = {best['n_estimators']}")
    print(f"max_depth = {best['max_depth']}")
    print(f"min_samples_split = {best['min_samples_split']}")
    print(f"CV Accuracy = {best['CV_Accuracy']}")
    print(f"Gap = {best['Gap']}")

    with open(
        "outputs/final_analysis.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("TREE OF THOUGHT ANALYSIS\n\n")

        f.write(
            f"Best Configuration: {best['Config']}\n"
        )

        f.write(
            f"n_estimators={best['n_estimators']}\n"
        )

        f.write(
            f"max_depth={best['max_depth']}\n"
        )

        f.write(
            f"CV Accuracy={best['CV_Accuracy']}\n"
        )

    print(
        "\nAnalysis saved to outputs/final_analysis.txt"
    )


if __name__ == "__main__":
    analyze_results()