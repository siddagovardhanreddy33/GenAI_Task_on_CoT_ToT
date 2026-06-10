import pandas as pd
from analysis.sentiment_analyzer import analyze_review

with open("prompts/cot_prompt.txt","r",encoding="utf-8") as f:
    template = f.read()

df = pd.read_csv("data/reviews.csv")
results = []

for review in df["review"]:
    prompt = template.replace("{review}",review)
    analysis = analyze_review(prompt)

    results.append({"review": review,"analysis": analysis})

result_df = pd.DataFrame(results)

result_df.to_csv("outputs/sentiment_results.csv",index=False)
print("Analysis Complete!")