import pandas as pd
import matplotlib.pyplot as plt


# Load BKT history
df = pd.read_csv("data/bkt_history.csv")


# Create the plot
plt.figure(figsize=(10, 6))


# Plot each skill
for skill in df["skill"].unique():

    skill_data = df[df["skill"] == skill]

    plt.plot(
        skill_data["question_id"],
        skill_data["mastery_after"],
        marker="o",
        label=skill
    )


# Labels
plt.xlabel("Question")
plt.ylabel("Mastery Probability")
plt.title("BKT Mastery Trajectory")


# Mastery probability is between 0 and 1
plt.ylim(0, 1)


# Legend
plt.legend()


# Make question labels readable
plt.xticks(rotation=45)


plt.tight_layout()


# Save the graph
plt.savefig("data/bkt_mastery_trajectory.png")


# Display the graph
plt.show()