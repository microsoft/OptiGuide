from pathlib import Path
import json
import re
import sys
import math

def average_turn_accuracies(root_dir: str, turns=None, use_new: bool = False):
    """
    Compute average accuracies and std dev for turn_0..turn_4 over subdirectories named 'xxx_j'.
    Returns (averages_dict, stdevs_dict, counts_dict).

    If use_new is True, read from stats_new.json instead of stats.json.
    """
    if turns is None:
        turns = [f"turn_{i}" for i in range(5)]  # turn_0 .. turn_4

    subdir_pat = re.compile(r".+_\d+$")
    sums = {t: 0.0 for t in turns}
    sumsq = {t: 0.0 for t in turns}
    counts = {t: 0 for t in turns}

    stats_filename = "stats_new.json" if use_new else "stats.json"

    for d in Path(root_dir).iterdir():
        if not d.is_dir() or not subdir_pat.match(d.name):
            continue
        stats_path = d / stats_filename
        if not stats_path.exists():
            continue
        try:
            data = json.loads(stats_path.read_text())
        except Exception:
            continue
        acc = data.get("accuracy_per_turn", {})
        for t in turns:
            v = acc.get(t)
            if isinstance(v, (int, float)):
                v = float(v)
                sums[t] += v
                sumsq[t] += v * v
                counts[t] += 1

    averages = {t: (sums[t] / counts[t] if counts[t] else None) for t in turns}
    stdevs = {}
    for t in turns:
        n = counts[t]
        if n > 1:
            # sample standard deviation (ddof=1)
            var = (sumsq[t] - (sums[t] ** 2) / n) / (n - 1)
            # guard tiny negative from FP error
            stdevs[t] = math.sqrt(var) if var > 0 else 0.0
        elif n == 1:
            stdevs[t] = 0.0
        else:
            stdevs[t] = None

    return averages, stdevs, counts

def main():
    prog = Path(sys.argv[0]).name
    args = sys.argv[1:]

    if not args:
        print(f"Usage: {prog} [--new] <root_dir>")
        sys.exit(1)

    use_new = False
    if "--new" in args:
        use_new = True
        args.remove("--new")

    if not args:
        print(f"Usage: {prog} [--new] <root_dir>")
        sys.exit(1)

    root_dir = args[0]
    averages, stdevs, counts = average_turn_accuracies(root_dir, use_new=use_new)

    # Print sorted by turn index
    for t in sorted(averages, key=lambda k: int(k.split('_')[1])):
        avg = averages[t]
        std = stdevs[t]
        n = counts[t]

        if avg is None:
            print(f"{t}: None (std=None) (n={n})")
        else:
            avg *= 100.0
            std_str = "None" if std is None else f"{std * 100.0:.6f}"
            print(f"{t}: {avg:.6f} ± {std_str} (n={n})")

if __name__ == "__main__":
    main()
