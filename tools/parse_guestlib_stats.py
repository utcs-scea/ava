#!/usr/bin/python3

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, type=str)
    args = parser.parse_args()

    stats = {}

    with open(args.file) as fin:
        for line in fin:
            sp = line.strip().split(",")
            time = sp[1].strip()
            namesp = sp[0].split(" ")
            name = namesp[1]
            if name not in stats:
                stats[name] = [0, []]
            stats[name][0] += 1
            stats[name][1].append(int(time))
    title = [
        "name".ljust(40),
        "called times".ljust(12),
        "p50(us)".ljust(10),
        "p90(us)".ljust(10),
        "p99(us)".ljust(10),
        "total(ms)".ljust(10),
    ]
    print("|".join(title))
    keys = sorted(stats.keys())
    for n in keys:
        exec_time = np.array(stats[n][1])
        p50 = np.percentile(exec_time, 50) / 1000.0
        p90 = np.percentile(exec_time, 90) / 1000.0
        p99 = np.percentile(exec_time, 99) / 1000.0
        total = np.sum(exec_time) / 1000000.0
        data = [
            str(n).ljust(40),
            str(stats[n][0]).ljust(12),
            str(round(p50, 2)).ljust(10),
            str(round(p90, 2)).ljust(10),
            str(round(p99, 2)).ljust(10),
            str(round(total, 3)).ljust(10),
        ]
        line = "|".join(data)
        print(line)


if __name__ == "__main__":
    main()
