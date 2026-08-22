"""Regression check: the supplement must read the same in both builds.

heterogeneous/ produces three PDFs from two shared bodies:

    heterogeneous.pdf   the paper alone
    supplement.pdf      the supplement alone
    submission.pdf      both in one file

The supplement's references to the paper go through \\paperref and \\paperrefii,
which the two shells define differently -- literal text in the standalone build,
real \\ref in the combined one. That is exactly the kind of construction that can
silently diverge: a macro whose combined definition drops an argument prints a
different sentence in each PDF and nothing errors. It happened once, turning
"Theorems 2 and 3" into "Theorems 2" in the combined build only.

So compare the rendered supplement text between the two PDFs. Any difference is
a bug in a shell definition, not in the body.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent.parent
# "heterogeneous/" in the working repository, "paper/" in the replication package
HERE = next((ROOT / d for d in ("heterogeneous", "paper")
             if (ROOT / d / "submission.pdf").exists()), None)
START = "This supplement reports"


def text(path):
    """Page text with the page-number line dropped.

    Page numbers legitimately differ between builds (1-3 standalone, 17-19
    combined) and sit on their own line at the foot of each page. Strip exactly
    those, and nothing else: a blanket rule against isolated digits would also
    erase the numbers in "Proposition 2", which are the whole point of the check.
    """
    out = []
    for page in pypdf.PdfReader(path).pages:
        lines = [l for l in (page.extract_text() or "").split("\n") if l.strip()]
        if lines and lines[-1].strip().isdigit():
            lines.pop()
        out.append("\n".join(lines))
    return "\n".join(out)


# The standalone supplement carries its own \maketitle, hence the affiliation
# footnote; the combined build does not. A structural difference, not a bug.
FOOTNOTE = re.compile(r"[\u2217*]\s*Dept\..*?rvicente@ime\.usp\.br", re.S)


def normalise(s):
    """Collapse whitespace, drop the title footnote, unify the S-prefix."""
    s = FOOTNOTE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bS(\d)", r"\1", s)
    # Drop every hyphen and any space following it. Line breaks fall in different
    # places in the two builds, so the same word may be hyphenated in one and not
    # the other; applying the rule to both makes the comparison blind to it.
    s = re.sub(r"-\s*", "", s)
    return s.strip()


def supplement_body(path):
    t = text(path)
    if START not in t:
        sys.exit(f"{path.name}: could not find the supplement (looked for {START!r})")
    return normalise(t[t.index(START):])


def main():
    if HERE is None:
        sys.exit("could not find submission.pdf under heterogeneous/ or paper/")
    a = supplement_body(HERE / "supplement.pdf")
    b = supplement_body(HERE / "submission.pdf")

    # Compare word multisets rather than sequences. Float placement legitimately
    # differs between builds -- a table can land either side of a heading -- and
    # that reorders words without changing which words are present. A macro that
    # drops an argument does change them, which is what this guards against.
    ca, cb = Counter(a.split()), Counter(b.split())
    if ca == cb:
        print(f"OK  the supplement renders the same words in both builds "
              f"({sum(ca.values())} words)")
        return 0

    print("MISMATCH between supplement.pdf and the supplement in submission.pdf\n")
    for label, diff in (("only in the standalone build", ca - cb),
                        ("only in the combined build", cb - ca)):
        if diff:
            print(f"  {label}:")
            for w, n in sorted(diff.items()):
                print(f"    {n}x {w!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
