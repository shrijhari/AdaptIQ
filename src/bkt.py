class BKT:
    def __init__(
        self,
        initial_mastery,
        p_learn=0.20,
        p_slip=0.05,
        p_guess=0.20
    ):
        self.mastery = initial_mastery.copy()
        self.p_learn = p_learn
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update_skill(self, skill, correct):
        p_l = self.mastery[skill]

        if correct == 1:
            # Probability of getting the question correct
            p_correct = (
                p_l * (1 - self.p_slip)
                + (1 - p_l) * self.p_guess
            )

            # Probability that the student knew the skill
            # given that they answered correctly
            p_known = (
                p_l * (1 - self.p_slip)
            ) / p_correct

        else:
            # Probability of getting the question wrong
            p_incorrect = (
                p_l * self.p_slip
                + (1 - p_l) * (1 - self.p_guess)
            )

            # Probability that the student knew the skill
            # given that they answered incorrectly
            p_known = (
                p_l * self.p_slip
            ) / p_incorrect

        # Learning transition
        p_new = p_known + (1 - p_known) * self.p_learn

        self.mastery[skill] = p_new

        return p_new

if __name__ == "__main__":

    initial_mastery = {
        "S1": 0.50,
        "S2": 0.40,
        "S3": 0.20,
        "S4": 0.60,
        "S5": 0.70,
        "S6": 0.10
    }

    bkt = BKT(initial_mastery)

    print("Initial mastery:")
    print(bkt.mastery)

    print("\nS3 - Correct:")
    print(bkt.update_skill("S3", 1))

    print("\nS3 - Correct again:")
    print(bkt.update_skill("S3", 1))

    print("\nS3 - Wrong:")
    print(bkt.update_skill("S3", 0))
    