import json


def format_json(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(format_json(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = (json.dumps(key) + ": " + format_json(item) for key, item in value.items())
        return "{" + ", ".join(fields) + "}"
    raise TypeError(type(value))


with open("answers.json", encoding="utf-8") as source:
    answers = json.load(source)

# Preserve the PDF's scalar JSON templates. All scalar statistics are defined,
# so the template's undefined-value helper is null.
helper_after = {
    "E4": "total_distance",
    "E5": "top_rate",
    "E6": "median_walkdistance",
    "E7": "mean_headshot_share",
    "E13": "longestkill",
    "E14": "fraction",
    "E15": "mean_swim_distance",
    "E17": "total_damage",
    "E22": "fraction",
    "E23": "q1",
    "E24": "fraction",
    "E25": "median_ride_distance",
    "M1": "player_mean",
    "M2": "mean_concentration",
    "M3": "mean_damage_percentile",
    "M15": "pooled_player_mean",
    "M18": "maximum_grid_distance",
    "M20": "same_fraction",
    "M23": "mean_assists_per_kill",
    "M24": "damage_auc",
    "H2": "match_mae_a",
    "H3": "extra_0_allocation",
    "H4": "observed_mean_difference",
    "H5": "intercept",
}

for question, after_key in helper_after.items():
    updated = {}
    for key, value in answers[question].items():
        updated[key] = value
        if key == after_key:
            updated["use_undefined_if_not_defined"] = None
    answers[question] = updated

with open("submission_answers.txt", "w", encoding="utf-8", newline="\n") as output:
    for question, answer in answers.items():
        output.write(f"{question}: {format_json(answer)}\n")

with open("copy_paste_answers.md", "w", encoding="utf-8", newline="\n") as output:
    for question, answer in answers.items():
        output.write(f"## {question}\n\n")
        output.write("```json\n")
        output.write(format_json(answer) + "\n")
        output.write("```\n\n")
