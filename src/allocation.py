def allocate_questions(mastery, n_questions, available=None, blend=0.7):
    """Split n_questions across skills: more for weak skills, fewer for strong
    ones, but every skill is covered once n_questions >= number of skills.

    mastery:   {"S1": 0.9, ...}
    available: skills that actually have questions (default: all)
    blend:     0..1  how strongly weakness drives the split (0 = even split).
               <1 keeps plans from over-reacting to noisy diagnostic scores.
    """
    skills = [s for s in mastery if available is None or s in available]
    if not skills or n_questions <= 0:
        return {}

    # Fewer questions than skills: cover the weakest skills, one each.
    if n_questions <= len(skills):
        weakest = sorted(skills, key=lambda s: mastery[s])[:n_questions]
        return {s: (1 if s in weakest else 0) for s in skills}

    plan = {s: 1 for s in skills}                      # coverage floor
    extra = n_questions - len(skills)

    need = {s: blend * (1 - mastery[s]) + (1 - blend) * 0.5 for s in skills}
    total = sum(need.values())
    share = {s: extra * need[s] / total for s in skills}

    for s in skills:                                   # whole parts first
        plan[s] += int(share[s])
    left = n_questions - sum(plan.values())            # then largest remainders
    for s in sorted(skills, key=lambda s: share[s] - int(share[s]), reverse=True)[:left]:
        plan[s] += 1
    return plan

if __name__ == "__main__":
    m = {"S1": .9, "S2": .3, "S3": .5, "S4": .7, "S5": .2, "S6": .6}
    for n in (4, 6, 12, 30):
        p = allocate_questions(m, n)
        print(n, p, "sum =", sum(p.values()))
    print("S6 has no content, n=12:", allocate_questions(m, 12, available={"S1","S2","S3","S4","S5"}))
    print("everyone equal, n=12:", allocate_questions({f"S{i}": .5 for i in range(1,7)}, 12))
