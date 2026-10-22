#!/usr/bin/env python
"""
German text normalization helpers for entity matching.

Rule-based and dependency-free. Shared by the entity resolution in
botIntents.py (and reusable elsewhere, e.g. botActions.py).

The stemmer deliberately implements only what a closed 162-name
vocabulary needs: umlaut stem mapping and German plural suffix
stripping. It is symmetric (input tokens and entity names go through
the same function), so e.g. "Frösche" and "Frosch" reduce to the
same stem while "Windröschen" is protected by its diminutive
"-chen" ending and stays distinguishable.
"""

from __future__ import annotations

import re

# Order matters: most specific plural suffix first.
# No "-s": stripping it loops words like "Wiesen" down to "wie"
# (a function-word collision); -s differences between input and
# item names are covered by stage-1 prefix/suffix containment.
_PLURAL_SUFFIXES = ("ern", "en", "er", "e", "n")
_MIN_STEM_LEN = 3

_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def german_stem(word: str) -> str:
    """Reduce a German word to a crude plural-insensitive stem.

    Lowercases, maps umlauts to their base vowels (ä→a, ö→o, ü→u)
    and strips plural suffixes (most specific first, repeatedly until
    stable), never leaving a stem shorter than 3 characters. Words
    ending in the diminutive "-chen" are returned unchanged (after
    lowercasing and umlaut mapping) so e.g. "Windröschen" does not
    lose its ending.

    The reduction runs to stability, which makes the function
    idempotent: stemming an already-stemmed word returns the same
    stem. Because input tokens and entity names go through the same
    function, aggressive stripping stays symmetric and cannot
    desynchronize a match.
    """
    w = (word or "").lower()
    w = w.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    if w.endswith("chen"):
        return w
    changed = True
    while changed:
        changed = False
        for suffix in _PLURAL_SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM_LEN:
                w = w[: -len(suffix)]
                changed = True
                break
    return w


def normalize(text: str) -> list:
    """Tokenize text on non-letters and return the stem of each token."""
    return [german_stem(tok) for tok in _TOKEN_RE.findall(text or "")]


# Stems of German function words. Only used to guard loose
# prefix/suffix containment for short (3-char) stems, where words
# like "wie" would otherwise spuriously match "Wiesen-..." stems.
FUNCTION_WORD_STEMS = frozenset({
    "aber", "alle", "alles", "als", "auch", "auf", "aus", "bei", "bin",
    "bis", "bist", "da", "damit", "dann", "das", "dass", "dein", "dem",
    "den", "denn", "der", "des", "die", "dir", "du", "ein", "eine",
    "einem", "einen", "er", "es", "euch", "gab", "gib", "gibt", "hab",
    "habe", "haben", "hat", "hast", "hatte", "hier", "ich", "ihr", "im",
    "in", "ist", "ja", "je", "jemand", "jetzt", "kann", "kein", "koennte",
    "mal", "man", "mein", "mir", "mit", "muss", "nach", "nein", "nicht",
    "noch", "nur", "ob", "oder", "ohne", "seid", "sein", "sich", "sie",
    "sind", "so", "soll", "sonst", "und", "uns", "unser", "von", "vor",
    "wann", "war", "warum", "was", "weg", "weil", "weiter", "welche",
    "wer", "wie", "wir", "wird", "wo", "wollen", "wird", "wuerde", "zu",
    "zum", "zur",
})


if __name__ == "__main__":
    samples = ["Frösche", "Frosch", "Windröschen", "Moorfrosch", "Bäume"]
    for s in samples:
        stem = german_stem(s)
        assert german_stem(stem) == stem, f"not idempotent: {s} -> {stem}"
        print(f"{s!r} -> {stem!r}")
