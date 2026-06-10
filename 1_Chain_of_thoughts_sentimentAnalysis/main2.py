from analysis.sentiment_analyzer import analyze_review

review = """This phone has amazing battery life but gets hot while gaming."""

with open("prompts/cot_prompt.txt","r",encoding="utf-8") as f:
    template = f.read()

prompt = template.replace("{review}", review)
print(analyze_review(prompt))