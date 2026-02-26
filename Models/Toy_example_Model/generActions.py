#!/usr/bin/env python3
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

# =====================
# PARAMÈTRE GLOBAL
# =====================
EPSILON = 0.05   # proba minimale par sous-action

def compositions(total: int, k: int):
    """Génère toutes les k-compositions faibles de total."""
    for bars in combinations(range(total + k - 1), k - 1):
        xs = []
        prev = -1
        for b in bars + (total + k - 1,):
            xs.append(b - prev - 1)
            prev = b
        yield xs

def main():
    ap = argparse.ArgumentParser(
        description="Génère des actions (vecteurs de probas) avec p_i >= epsilon."
    )
    ap.add_argument("--k", type=int, required=True, help="Nombre de sous-actions (K >= 1).")
    ap.add_argument("--den", type=int, default=10,
                    help="Résolution pour la masse libre (plus grand = plus fin).")
    ap.add_argument("--out", type=Path, default=Path("actions.txt"))
    ap.add_argument("--fmt", type=str, default="{:.6f}")
    args = ap.parse_args()

    k = args.k
    den = args.den

    if k < 1:
        raise SystemExit("k doit être >= 1")

    if EPSILON * k > 1.0 + 1e-12:
        raise SystemExit("Erreur: K * epsilon > 1 (impossible)")

    # --- Cas K = 1 ---
    if k == 1:
        args.out.write_text("1\n0 1.0\n", encoding="utf-8")
        print(f"Wrote 1 action to {args.out} (k=1)")
        return

    # Masse libre à distribuer
    R = 1.0 - k * EPSILON
    if R < 0:
        raise SystemExit("Erreur: masse libre négative")

    # On discrétise la masse libre avec 'den'
    # r_i = x_i / den * R
    total = den

    lines = []
    action_id = 0

    for xs in compositions(total, k):
        rs = [x / den * R for x in xs]
        ps = [EPSILON + r for r in rs]

        # Sécurité numérique
        s = sum(ps)
        ps = [p / s for p in ps]

        probs_str = " ".join(args.fmt.format(p) for p in ps)
        lines.append(f"{action_id} {probs_str}")
        action_id += 1

    content = [str(action_id)] + [str(k)] + lines
    args.out.write_text("\n".join(content) + "\n", encoding="utf-8")

    print(f"Wrote {action_id} actions to {args.out} (k={k}, epsilon={EPSILON})")

if __name__ == "__main__":
    main()
