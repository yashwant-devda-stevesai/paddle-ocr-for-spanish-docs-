
"""
post_processing.py

Usage (standalone CLI)
    python post_processing.py --input result_1.json
    python post_processing.py --input result_1.json --output cleaned.txt
    python post_processing.py --input ./output_pdf/ --output ./cleaned/

"""

import re
import json
import unicodedata
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter, defaultdict

from symspellpy import SymSpell, Verbosity
from wordfreq import top_n_list, zipf_frequency
from unidecode import unidecode


LAYOUT_LABELS: frozenset = frozenset({
    "text", "header", "footer", "number", "footnote",
    "header_image", "footer_image", "aside_text", "caption",
    "title", "doc_title", "paragraph_title", "paragraph",
    "table_caption", "formula_number", "image", "table",
    "equation", "figure", "figure_caption", "reference",
    "abstract", "section",
})

SKIP_LABELS: frozenset = frozenset({
    "footer",
    "footer_image",
    "header_image",
})

DEDUPE_LABELS: frozenset = frozenset({"header"})

_LABEL_LINE_RE = re.compile(
    r"^(" + "|".join(re.escape(l) for l in LAYOUT_LABELS) + r")$",
    re.IGNORECASE,
)
_FILEPATH_LINE_RE = re.compile(r"^.{1,400}\.(pdf|PDF|txt|TXT)$")

_LATEX_REPLACEMENTS = [
    (r"\$\s*N\s*\^\s*\{\\circ\}\s*\$",              "N\u00b0"),
    (r"\$\s*n\s*\^\s*\{\\circ\}\s*\$",              "n\u00b0"),
    (r"\$\s*N\^\\circ\s*\$",                         "N\u00b0"),
    (r"\\circ",                                       "\u00b0"),
    (r"\$\s*\\underline\{\\text\{([^}]*)\}\}\s*\$",  r"\1"),
    (r"\$\s*\\underline\{([^}]*)\}\s*\$",            r"\1"),
    (r"(\$\s*\\frac\{1\}\{2\}\s*\$[\s\n]*)+",        ""),
    (r"\$\s*\\frac\{[^}]*\}\{[^}]*\}\s*\$",          ""),
    (r"\\text\{([^}]*)\}",                            r"\1"),
    (r"\\textbf\{([^}]*)\}",                          r"\1"),
    (r"\\textit\{([^}]*)\}",                          r"\1"),
    (r"\\emph\{([^}]*)\}",                            r"\1"),
    (r"\^\{([^}]*)\}",                                r"\1"),
    (r"_\{([^}]*)\}",                                 r"\1"),
    (r"\$\$[^$]*\$\$",                               ""),
    (r"\$[^$\n]{0,80}\$",                            ""),
    (r"\\[a-zA-Z]+\{[^}]*\}",                        ""),
    (r"\\[a-zA-Z]+",                                  ""),
]
_LATEX_COMPILED = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), rep)
    for pat, rep in _LATEX_REPLACEMENTS
]

# Non-Latin script detection (hallucination removal)
_NON_LATIN_RE = re.compile(
    "["
    "\u0400-\u04FF"   # Cyrillic
    "\u0600-\u06FF"   # Arabic
    "\u0900-\u097F"   # Devanagari
    "\u3000-\u9FFF"   # CJK, Hiragana, Katakana
    "\uAC00-\uD7AF"   # Hangul
    "\uF900-\uFAFF"   # CJK compatibility
    "]"
)
_NON_LATIN_THRESHOLD = 0.30
_CJK_DATE_RE = re.compile(
    r"\d{4}\u5e74\d{1,2}\u6708\d{1,2}\u65e5"
)

_GARBAGE_PATTERNS = [
    re.compile(r"^[^\w\s\u00C0-\u024F]{4,}$"),   
    re.compile(r"^(.)\1{5,}$"),                   
    re.compile(r"^\s*[\.\-_=~]{3,}\s*$"),          
]

_LINEBREAK_HYPHEN_RE = re.compile(
    r"([A-Za-z\u00C0-\u024F])-[ \t]*\n[ \t]*([A-Za-z\u00C0-\u024F])",
    re.MULTILINE | re.UNICODE,
)

_INLINE_CAPS_HYPHEN_RE = re.compile(
    r"\b([A-Z\u00C0-\u024F]{2,})-([A-Z\u00C0-\u024F]{2,})\b"
)

_OCR_SUBS = [
    (re.compile(r"(?<=[a-z\u00e0-\u00ff])0(?=[a-z\u00e0-\u00ff])", re.I), "o"),
    (re.compile(r"(?<=[a-z\u00e0-\u00ff])1(?=[a-z\u00e0-\u00ff])", re.I), "l"),
    (re.compile(r"\brn(?=[a-z])", re.I), "m"),
    (re.compile(r"ii", re.I), "ll"),
]


_MISSING_SPACE_RE = re.compile(r"([.;,])([A-Z\u00C0-\u024F])")
_LETTER_DIGIT_SPACE_RE = re.compile(r"([a-z\u00E0-\u00FF])(\d)", re.UNICODE)

_ALPHA_ONLY_RE = re.compile(r"[^\wÁÉÍÓÚÑáéíóúñ]")
_LETTER_HYPHEN_LETTER_RE = re.compile(r"[A-Za-z\u00C0-\u024F]-[A-Za-z\u00C0-\u024F]")
_HYPHENATED_ALPHA_TOKEN_RE = re.compile(
    r"\b([A-Za-z\u00C0-\u024F]{2,})-([A-Za-z\u00C0-\u024F]{2,})\b"
)
_INTERNAL_SEPARATOR_RE = re.compile(r"(?<=\w)[^\w\s]+(?=\w)", re.UNICODE)
_ROMAN_TOKEN_RE = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})"
    r"(IX|IV|V?I{0,3})$"
)


_SYMSPELL_TOP_N = 150000
_sym_spell = SymSpell(max_dictionary_edit_distance=2)
for _w in top_n_list("es", _SYMSPELL_TOP_N):
    _sym_spell.create_dictionary_entry(_w, 1)


_NUM_FRAC_RE  = re.compile(r"^[\d.,/:%()\-+]+$")
_USD_RE       = re.compile(r"^U[.]S[.]\$?$")

_FINANCIAL_COMPOUND_SUFFIXES = ("dolares", "euros", "francos", "libras")


def _looks_like_roman_token(token: str) -> bool:
    clean = _ALPHA_ONLY_RE.sub("", token)
    if len(clean) < 2:
        return False
    norm = clean.replace("l", "I").upper()
    return bool(_ROMAN_TOKEN_RE.match(norm))


def _is_frozen(token: str) -> bool:
    clean = _ALPHA_ONLY_RE.sub("", token)
    if not clean:
        return True
    if len(clean) <= 2:
        return True
    if _looks_like_roman_token(clean):
        return True
    if clean.isupper():
        return True
    if _NUM_FRAC_RE.match(token):
        return True
    if _USD_RE.match(token):
        return True
    if token.startswith("$") and len(token) > 1:
        return True
    if "\u00b0" in token:      
        return True
    norm = unidecode(clean.lower())
    for suffix in _FINANCIAL_COMPOUND_SUFFIXES:
        if norm.endswith(suffix) and len(norm) > len(suffix) + 2:
            return True
    return False


def _strip_trailing_schema(lines: list) -> list:
    last_real = -1
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if s and not _LABEL_LINE_RE.match(s):
            last_real = i
            break
    return lines if last_real == -1 else lines[: last_real + 1]


def _remove_filepath_line(lines: list) -> list:
    if not lines:
        return lines
    first = lines[0].strip()
    if (
        _FILEPATH_LINE_RE.match(first)
        or "\\" in first
        or (first.startswith("/") and "." in first)
    ):
        return lines[1:]
    return lines


def _remove_initial_label_block(lines: list) -> list:
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or _LABEL_LINE_RE.match(s):
            i += 1
        else:
            break
    return lines[i:]


def _remove_label_lines(lines: list) -> list:
    return [l for l in lines if not (l.strip() and _LABEL_LINE_RE.match(l.strip()))]


def _remove_non_latin_lines(lines: list) -> list:
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if _CJK_DATE_RE.search(s):
            continue
        non_latin = len(_NON_LATIN_RE.findall(s))
        total = len([c for c in s if not c.isspace()])
        if total > 0 and (non_latin / total) > _NON_LATIN_THRESHOLD:
            continue
        result.append(line)
    return result


def _remove_garbage_lines(lines: list) -> list:
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if any(p.match(s) for p in _GARBAGE_PATTERNS):
            continue
        result.append(line)
    return result


def _remove_single_char_junk(lines: list) -> list:
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if len(s) == 1 and not s.isalnum():
            continue
        if len(s) == 1 and s.isupper():
            continue
        result.append(line)
    return result


def _collapse_blank_lines(lines: list) -> list:
    result = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return result


def _apply_latex_fixes(text: str) -> str:
    for pattern, replacement in _LATEX_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def _join_linebreak_hyphens(text: str) -> str:
    return _LINEBREAK_HYPHEN_RE.sub(lambda m: m.group(1) + m.group(2), text)


def _join_caps_hyphens(text: str) -> str:
    return _INLINE_CAPS_HYPHEN_RE.sub(
        lambda m: m.group(1) + m.group(2), text
    )


def _join_dictionary_hyphens(text: str) -> str:
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if token.isupper():
            return token

        left, right = token.split("-", 1)
        if min(len(left), len(right)) > 4:
            return token

        joined = left + right
        left_norm = unidecode(left.lower())
        right_norm = unidecode(right.lower())
        if (
            _sym_spell.lookup(left_norm, Verbosity.TOP, max_edit_distance=0)
            and _sym_spell.lookup(right_norm, Verbosity.TOP, max_edit_distance=0)
            and zipf_frequency(left_norm, "es") >= 2.5
            and zipf_frequency(right_norm, "es") >= 2.5
        ):
            return token

        norm = unidecode(joined.lower())
        if zipf_frequency(norm, "es") >= 1.3:
            return joined
        if _sym_spell.lookup(norm, Verbosity.TOP, max_edit_distance=0):
            return joined
        return token

    return _HYPHENATED_ALPHA_TOKEN_RE.sub(_replace, text)


def _split_trailing_conjunction(token: str) -> str:
    if _is_frozen(token):
        return token

    clean = _ALPHA_ONLY_RE.sub("", token)
    if len(clean) < 5:
        return token

    norm = unidecode(clean.lower())
    if len(norm) > 6 and norm.endswith(("ase", "ese")):
        return token

    if zipf_frequency(norm, "es") > 0:
        return token

    if _sym_spell.lookup(norm, Verbosity.TOP, max_edit_distance=0):
        return token

    for conj in ("y", "o", "e"):
        if norm.endswith(conj) and len(norm) > len(conj) + 3:
            stem_norm = norm[: -len(conj)]
            stem_valid = _sym_spell.lookup(stem_norm, Verbosity.TOP, max_edit_distance=0)
            if stem_valid and zipf_frequency(stem_norm, "es") > 1.5:
                stem_original = clean[: -len(conj)]
                return f"{stem_original} {conj}"

    return token


def _split_merged_token(token: str) -> str:
    if _is_frozen(token):
        return token
    if _INTERNAL_SEPARATOR_RE.search(token):
        return token

    clean = _ALPHA_ONLY_RE.sub("", token)
    if not clean:
        return token
    if clean.isupper():
        return token
    if len(clean) < 8:
        return token

    norm = unidecode(clean.lower())

    if zipf_frequency(norm, "es") > 0:
        return token

    if _sym_spell.lookup(norm, Verbosity.TOP, max_edit_distance=0):
        return token
    near_valid = _sym_spell.lookup(norm, Verbosity.TOP, max_edit_distance=1)
    if near_valid and near_valid[0].distance <= 1:
        return token

    best_score = -1.0
    best_split = None

    for i in range(3, len(norm) - 2):
        left_n  = norm[:i]
        right_n = norm[i:]

        if not _sym_spell.lookup(left_n, Verbosity.TOP, max_edit_distance=0):
            continue
        if not _sym_spell.lookup(right_n, Verbosity.TOP, max_edit_distance=0):
            continue

        score = zipf_frequency(left_n, "es") + zipf_frequency(right_n, "es")
        if score > best_score:
            best_score = score
            best_split = (clean[:i], clean[i:])

    if best_split and best_score >= 6:
        left_part, right_part = best_split
        if len(right_part) >= 10:
            right_part = _split_merged_token(right_part)
        return f"{left_part} {right_part}"

    return token


def _split_merged_words(text: str) -> str:

    corrected_lines = []
    for line in text.split("\n"):
        tokens = line.split()
        new_tokens = []
        for t in tokens:
            t = _split_upper_dictionary_compound(t)
            t = _split_merged_token(t)
            sub = []
            for part in t.split():
                sub.append(_split_trailing_conjunction(part))
            new_tokens.append(" ".join(sub))
        corrected_lines.append(" ".join(new_tokens))
    return "\n".join(corrected_lines)


_LEGAL_PHRASE_FIXES = [
    (re.compile(r"\bSE\s+RESCUEVE\b"), "SE RESUELVE"),
    (re.compile(r"\bse\s+na\s+hecho\b", re.IGNORECASE), "se ha hecho"),
    (re.compile(r"\bdejando\s+a\s+salve\s+derechos\b", re.IGNORECASE),
     "dejando a salvo derechos"),
    (re.compile(r"\bPresidente\s+(Construccional|Construcciones)\s+de\s+la\s+República\b"),
     "Presidente Constitucional de la República"),
    (re.compile(r"\bExpidase\b"), "Expídase"),
    (re.compile(r"\bexpidase\b"), "expídase"),
    (re.compile(r"\barchiv[ae]se\b"), "archívese"),
    (re.compile(r"\bArchiv[ae]se\b"), "Archívese"),
    (re.compile(r"\bPubliquese\b"), "Publíquese"),
    (re.compile(r"\bpubliquese\b"), "publíquese"),
    (re.compile(r"\bartículos\s+fotograf[ií]as\s+en\s+general\b", re.IGNORECASE),
     "artículos fotográficos en general"),
    (re.compile(
        r"\ben\s+el\s+c[oó]lero\s+productos\s+qu[ií]nios\s+para\s+"
        r"fotograf[ií]a\s+y\s+art[ií]culos\s+fotograficos\b",
        re.IGNORECASE,
    ), "en el comercio productos químicos para fotografía y artículos fotográficos"),
    (re.compile(r"\bque\s+se\s+elabora\s+o\s+(fabricar|fábrica)\s+en\b", re.IGNORECASE),
     "que se elabora o fabrica en"),
]

_QUOTED_TEXT_RE = re.compile(r"([\"“])([^\"”\n]{3,100})([\"”])")
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+")
_UPPER_NAME_PAIR_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ]{4,})\s+([A-ZÁÉÍÓÚÑ]{4,})\b"
)
_SPANISH_DATE_RE = re.compile(
    r"\b(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|"
    r"Octubre|Noviembre|Diciembre)\s+(\d{1,2})\s+de\s+(\d{4})\b"
)
_DATE_10XX_YEAR_RE = re.compile(
    r"\b(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|"
    r"Octubre|Noviembre|Diciembre)\s+(\d{1,2})\s+de\s+10(\d{2})\b"
)
_SPANISH_DAY_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+de\s+(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|"
    r"Agosto|Septiembre|Octubre|Noviembre|Diciembre)\s+de\s+(\d{4})\b"
)
_DAY_MONTH_10XX_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s+de\s+(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|"
    r"Agosto|Septiembre|Octubre|Noviembre|Diciembre)\s+de\s+10(\d{2})\b"
)


def _edit_distance_lte_one(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1

    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = j = edits = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _norm_word(word: str) -> str:
    return unidecode(word.lower())


def _apply_legal_phrase_fixes(text: str) -> str:
    for pattern, replacement in _LEGAL_PHRASE_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _apply_quoted_entity_consistency(text: str) -> str:
    groups = defaultdict(Counter)
    earliest = {}

    for match in _QUOTED_TEXT_RE.finditer(text):
        words = _WORD_RE.findall(match.group(2))
        if len(words) < 2 or len(words) > 6:
            continue
        suffix_key = tuple(_norm_word(w) for w in words[1:])
        first = words[0]
        groups[suffix_key][first] += 1
        earliest.setdefault((suffix_key, first), match.start())

    canonical = {}
    for suffix_key, counts in groups.items():
        if len(counts) < 2:
            continue
        best, best_count = min(
            counts.items(),
            key=lambda item: (-item[1], earliest[(suffix_key, item[0])]),
        )
        if best_count < 2:
            continue
        for variant in counts:
            if variant == best:
                continue
            if _edit_distance_lte_one(_norm_word(variant), _norm_word(best)):
                canonical[(suffix_key, variant)] = best

    if not canonical:
        return text

    def _replace(match: re.Match) -> str:
        quote_open, content, quote_close = match.groups()
        words = _WORD_RE.findall(content)
        if len(words) < 2 or len(words) > 6:
            return match.group(0)
        suffix_key = tuple(_norm_word(w) for w in words[1:])
        repl = canonical.get((suffix_key, words[0]))
        if not repl:
            return match.group(0)
        return quote_open + content.replace(words[0], repl, 1) + quote_close

    return _QUOTED_TEXT_RE.sub(_replace, text)


def _apply_upper_name_consistency(text: str) -> str:
    by_last = defaultdict(Counter)
    for first, last in _UPPER_NAME_PAIR_RE.findall(text):
        by_last[last][first] += 1

    replacements = {}
    for last, counts in by_last.items():
        if len(counts) < 2:
            continue
        for variant, variant_count in counts.items():
            if variant_count > 1:
                continue
            candidates = [
                (candidate, count)
                for candidate, count in counts.items()
                if count >= 2
                and candidate != variant
                and _edit_distance_lte_one(_norm_word(variant), _norm_word(candidate))
            ]
            if not candidates:
                continue
            canonical, _ = max(candidates, key=lambda item: item[1])
            replacements[(variant, last)] = (canonical, last)

    for (bad_first, last), (good_first, good_last) in replacements.items():
        text = re.sub(
            rf"\b{re.escape(bad_first)}\s+{re.escape(last)}\b",
            f"{good_first} {good_last}",
            text,
        )
    return text


def _apply_date_consistency(text: str) -> str:
    text = _DATE_10XX_YEAR_RE.sub(r"\1 \2 de 19\3", text)
    text = _DAY_MONTH_10XX_YEAR_RE.sub(r"\1 de \2 de 19\3", text)

    counts = Counter(_SPANISH_DATE_RE.findall(text))
    by_month_day = defaultdict(Counter)
    for month, day, year in counts:
        by_month_day[(month, day)][year] += counts[(month, day, year)]
    for day, month, year in _SPANISH_DAY_MONTH_DATE_RE.findall(text):
        by_month_day[(month, day)][year] += 1

    replacements = {}
    for (month, day), year_counts in by_month_day.items():
        repeated = [
            (year, count)
            for year, count in year_counts.items()
            if count >= 2
        ]
        if not repeated:
            continue
        for year, count in year_counts.items():
            if count > 1:
                continue
            candidates = [
                (candidate, candidate_count)
                for candidate, candidate_count in repeated
                if _edit_distance_lte_two(year, candidate)
            ]
            if not candidates:
                continue
            replacement, _ = max(candidates, key=lambda item: item[1])
            replacements[(month, day, year)] = replacement

    for (month, day, year), replacement in replacements.items():
        month_first_pattern = re.compile(rf"\b{month}\s+{day}\s+de\s+{year}\b")
        day_first_pattern = re.compile(rf"\b{day}\s+de\s+{month}\s+de\s+{year}\b")

        def _replace_date(match: re.Match) -> str:
            prefix = text[max(0, match.start() - 20):match.start()]
            if "Caduca" in prefix:
                return match.group(0)
            if match.group(0).startswith(month):
                return f"{month} {day} de {replacement}"
            return f"{day} de {month} de {replacement}"

        text = month_first_pattern.sub(_replace_date, text)
        text = day_first_pattern.sub(_replace_date, text)
    return text


def _edit_distance_lte_two(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= 0.5 and sum(
        c1 != c2 for c1, c2 in zip(a, b)
    ) <= 2


def _apply_contextual_legal_fixes(text: str) -> str:
    text = _apply_legal_phrase_fixes(text)
    text = _apply_quoted_entity_consistency(text)
    text = _apply_upper_name_consistency(text)
    text = _apply_date_consistency(text)
    return text



def _composite_score(norm_original: str, candidate_term: str) -> float:
    o, c = norm_original, candidate_term
    pos = sum(1 for a, b in zip(o, c) if a == b)
    seq = SequenceMatcher(None, o, c).ratio()
    freq = zipf_frequency(c, "es")
    return pos + seq * 2.0 + freq


_SINGULAR_DETERMINERS = frozenset({
    "el", "del", "al", "un", "una", "este", "esta", "ese", "esa",
    "aquel", "aquella", "cada",
})
_PLURAL_DETERMINERS = frozenset({
    "los", "las", "unos", "unas", "estos", "estas", "esos", "esas",
    "aquellos", "aquellas",
})

_AMOUNT_SEGMENTS = ()
_CURRENCY_SUFFIXES = ()

def _number_agreement_bonus(context_tokens: list, idx: int, candidate_term: str) -> float:
    if idx <= 0 or idx >= len(context_tokens):
        return 0.0

    prev = _ALPHA_ONLY_RE.sub("", context_tokens[idx - 1]).lower()
    if not prev:
        return 0.0

    cand = unidecode(candidate_term.lower())
    if len(cand) <= 3:
        return 0.0

    if prev in _SINGULAR_DETERMINERS:
        return 0.35 if not cand.endswith("s") else -0.35
    if prev in _PLURAL_DETERMINERS:
        return 0.35 if cand.endswith("s") else -0.35
    return 0.0


def _collapse_accent_variants(scored: list) -> list:
    best_by_norm = {}
    for sug, score in scored:
        key = unidecode(sug.term.lower())
        if key not in best_by_norm:
            best_by_norm[key] = (sug, score)
            continue

        prev_sug, prev_score = best_by_norm[key]
        if score > prev_score:
            best_by_norm[key] = (sug, score)
            continue
        if score == prev_score and sug.distance < prev_sug.distance:
            best_by_norm[key] = (sug, score)

    collapsed = list(best_by_norm.values())
    collapsed.sort(key=lambda x: x[1], reverse=True)
    return collapsed


def _segment_amount_prefix(prefix: str):
    n = len(prefix)
    dp = [None] * (n + 1)
    dp[0] = []

    for i in range(n):
        if dp[i] is None:
            continue
        for part in _AMOUNT_SEGMENTS:
            if prefix.startswith(part, i):
                j = i + len(part)
                cand = dp[i] + [part]
                if dp[j] is None or len(cand) < len(dp[j]):
                    dp[j] = cand
    return dp[n]


def _split_upper_amount_currency_compound(token: str) -> str:
    clean = _ALPHA_ONLY_RE.sub("", token)
    if not clean or not clean.isupper() or len(clean) < 12:
        return token

    norm = unidecode(clean.lower())
    for suffix in _CURRENCY_SUFFIXES:
        if not norm.endswith(suffix):
            continue

        prefix = norm[: -len(suffix)]
        if len(prefix) < 4:
            continue

        parts = _segment_amount_prefix(prefix)
        if not parts or len(parts) < 2:
            continue

        rebuilt = " ".join([*(p.upper() for p in parts), suffix.upper()])
        return token.replace(clean, rebuilt, 1)

    return token


def _split_upper_dictionary_compound(token: str) -> str:
    return token


def _correct_token(token: str, context_tokens: list, idx: int) -> str:
    clean = _ALPHA_ONLY_RE.sub("", token)
    if not clean or len(clean) <= 2 or clean.isdigit():
        return token
    norm = unidecode(clean.lower())

    if _is_frozen(token):
        return token
    if _INTERNAL_SEPARATOR_RE.search(token):
        return token

    if norm.endswith("aion") and len(norm) > 6:
        repaired = norm[:-4] + "acion"
        if (
            zipf_frequency(repaired, "es") >= 1.3
            or _sym_spell.lookup(repaired, Verbosity.TOP, max_edit_distance=0)
        ):
            return _apply_case(token, clean, repaired)


    if _LETTER_HYPHEN_LETTER_RE.search(token):
        return token

    if zipf_frequency(norm, "es") >= 1.3:
        return token


    if clean[0].isupper() and not clean.isupper():
        return token
    if len(norm) > 6 and norm.endswith(("ase", "ese")):
        return token

    if _sym_spell.lookup(norm, Verbosity.TOP, max_edit_distance=0):
        alts = _sym_spell.lookup(norm, Verbosity.ALL, max_edit_distance=1)
        current_score = _composite_score(norm, norm) + _number_agreement_bonus(
            context_tokens, idx, norm
        )
        for alt in alts:
            if alt.term == norm:
                continue
            alt_score = _composite_score(norm, alt.term) + _number_agreement_bonus(
                context_tokens, idx, alt.term
            )
            if alt_score > current_score + 1.5:
                return _apply_case(token, clean, alt.term)
        return token


    if len(norm) <= 3:
        max_dist = 1
    elif len(norm) <= 5:
        max_dist = 2
    else:
        max_dist = 2
    suggestions = _sym_spell.lookup(norm, Verbosity.ALL, max_edit_distance=max_dist)
    if not suggestions:
        return token

    scored = [
        (
            s,
            _composite_score(norm, s.term)
            + _number_agreement_bonus(context_tokens, idx, s.term),
        )
        for s in suggestions
    ]
    scored = _collapse_accent_variants(scored)

    best, best_score = scored[0]


    if len(scored) > 1:
        second, second_score = scored[1]
        margin = best_score - second_score
        min_margin = 0.05 if best.distance < second.distance else 0.25
        if margin < min_margin:
            return token   

    if best_score < 2.5:
        return token

    best_norm = unidecode(best.term.lower())
    similarity = SequenceMatcher(
        None,
        norm,
        best_norm
    ).ratio()

    if similarity < 0.80:
        return token

    return _apply_case(token, clean, best.term)

def _apply_case(original_token: str, clean: str, corrected: str):
    prefix_match = re.match(r"^\W*", original_token)
    suffix_match = re.search(r"\W*$", original_token)

    prefix = prefix_match.group() if prefix_match else ""
    suffix = suffix_match.group() if suffix_match else ""

    if original_token and original_token[0].isupper() and not original_token.isupper():
        corrected = corrected.capitalize()
    elif original_token.isupper():
        corrected = corrected.upper()

    return f"{prefix}{corrected}{suffix}"


def _correct_ocr_words(text: str) -> str:
    """Apply _correct_token to every token in the text."""
    corrected_lines = []
    for line in text.split("\n"):
        tokens = line.split()
        corrected_lines.append(
            " ".join(_correct_token(t, tokens, i) for i, t in enumerate(tokens))
        )
    return "\n".join(corrected_lines)



def _tokens_same_modulo_accents(a: str, b: str) -> bool:
    a_norm = unidecode(a.lower())
    b_norm = unidecode(b.lower())

    if len(a_norm) != len(b_norm):
        return False

    return a_norm == b_norm


def _build_accent_lookup(source_text: str) -> dict:

    lookup = {}
    for token in source_text.split():
        if _INTERNAL_SEPARATOR_RE.search(token):
            continue
        clean = _ALPHA_ONLY_RE.sub("", token)
        if not clean:
            continue
        if not any(c in clean for c in "áéíóúñÁÉÍÓÚÑ"):
            continue
        lookup[unidecode(clean.lower())] = clean
    return lookup


def _apply_accents_preserve_case(original_token: str, clean: str, accented: str) -> str:

    prefix_match = re.match(r"^\W*", original_token)
    suffix_match = re.search(r"\W*$", original_token)

    prefix = prefix_match.group() if prefix_match else ""
    suffix = suffix_match.group() if suffix_match else ""

    if clean.isupper():
        accented = accented.upper()
    elif clean.islower():
        accented = accented.lower()
    elif clean and clean[0].isupper() and clean[1:].islower():
        accented = accented.capitalize()

    return f"{prefix}{accented}{suffix}"


def validate_final_output(
    cleaned_text: str,
    blocks: list = None,
    raw_txt_text: str = None,
) -> str:

    if not cleaned_text:
        return cleaned_text

    raw_lookup  = _build_accent_lookup(raw_txt_text) if raw_txt_text else {}
    json_lookup = {}
    if blocks:
        for block in blocks:
            for token in block["text"].split():
                clean = _ALPHA_ONLY_RE.sub("", token)
                if clean and any(c in clean for c in "áéíóúñÁÉÍÓÚÑ"):
                    json_lookup[unidecode(clean.lower())] = clean

    validated_lines = []
    for line in cleaned_text.split("\n"):
        tokens = line.split()
        validated_tokens = []
        for token in tokens:
            if _INTERNAL_SEPARATOR_RE.search(token):
                validated_tokens.append(token)
                continue
            clean = _ALPHA_ONLY_RE.sub("", token)
            if not clean:
                validated_tokens.append(token)
                continue
            if _is_frozen(clean):
                validated_tokens.append(token)
                continue

            norm = unidecode(clean.lower())

            if norm in raw_lookup:
                raw_version = raw_lookup[norm]
                if _tokens_same_modulo_accents(clean, raw_version):
                    token = _apply_accents_preserve_case(token, clean, raw_version)
            elif norm in json_lookup:
                json_version = json_lookup[norm]
                if _tokens_same_modulo_accents(clean, json_version):
                    token = _apply_accents_preserve_case(token, clean, json_version)

            validated_tokens.append(token)
        validated_lines.append(" ".join(validated_tokens))

    return "\n".join(validated_lines)



def _fix_missing_spaces(text: str) -> str:

    text = _MISSING_SPACE_RE.sub(r"\1 \2", text)
    text = _LETTER_DIGIT_SPACE_RE.sub(r"\1 \2", text)
    return text


def _apply_ocr_subs(text: str) -> str:
    for pattern, replacement in _OCR_SUBS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_roman_tokens(text: str) -> str:
  
    lines_out = []
    for line in text.split("\n"):
        fixed = []
        for token in line.split():
            clean = _ALPHA_ONLY_RE.sub("", token)
            upper_count = sum(1 for c in clean if c.isupper())
            if (
                clean
                and _looks_like_roman_token(clean)
                and "l" in clean
                and upper_count >= 2
            ):
                token = token.replace(clean, clean.replace("l", "I"), 1)
            fixed.append(token)
        lines_out.append(" ".join(fixed))
    return "\n".join(lines_out)


def _normalize_whitespace(text: str) -> str:

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)



def extract_text_from_json(json_path: str) -> tuple:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_blocks = data.get("parsing_res_list", [])

    def _bbox_origin(block_obj: dict) -> tuple:
        bbox = block_obj.get("block_bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            try:
                return (float(bbox[1]), float(bbox[0]))
            except (TypeError, ValueError):
                pass
        return (1e9, 1e9)
    ordered_blocks = []

    for block in all_blocks:
        label   = block.get("block_label", "")
        content = block.get("block_content", "").strip()

        if not content:
            continue

        order = block.get("block_order")
        try:
            order = int(order) if order is not None else None
        except (TypeError, ValueError):
            order = None
        top_y, left_x = _bbox_origin(block)

        ordered_blocks.append({
            "text":  content,
            "label": label,
            "order": order,
            "score": block.get("score", 1.0),
            "_top_y": top_y,
            "_left_x": left_x,
        })

    ordered_tops = [
        b["_top_y"] for b in ordered_blocks
        if b["order"] is not None and b["_top_y"] < 1e9
    ]
    first_ordered_top = min(ordered_tops) if ordered_tops else 1e9

    def _row_bucket(top_y: float) -> int:
        if top_y >= 1e9:
            return 10**9
        return int(round(top_y / 25.0))

    def _block_sort_key(b: dict) -> tuple:
        if b["order"] is None:
            is_pre_body = b["_top_y"] <= first_ordered_top + 25
            group = 0 if is_pre_body else 2
            return (
                group,
                _row_bucket(b["_top_y"]),
                b["_left_x"],
                b["_top_y"],
                1e9,
            )
        return (1, b["order"], b["_top_y"], b["_left_x"], b["order"])

    ordered_blocks.sort(
        key=_block_sort_key
    )

    final_lines = []
    for block in ordered_blocks:
        text  = block["text"]
        label = block["label"]

        if label in {"header", "doc_title", "title"}:
            final_lines.append(text)
            final_lines.append("")
        elif label == "paragraph_title":
            final_lines.append("")
            final_lines.append(text)
            final_lines.append("")
        elif label == "paragraph":
            final_lines.extend(text.split("\n"))
            final_lines.append("")
        else:
            final_lines.append(text)
            final_lines.append("")

    raw_text = "\n".join(final_lines).strip()
    return ordered_blocks, raw_text



def normalize_page_text(raw_text: str, blocks: list = None) -> str:
    if not raw_text or not raw_text.strip():
        return ""

    text = _normalize_unicode(raw_text)

    text = _apply_latex_fixes(text)

    lines = text.splitlines()
    lines = _strip_trailing_schema(lines)
    lines = _remove_filepath_line(lines)
    lines = _remove_initial_label_block(lines)
    lines = _remove_label_lines(lines)

    lines = _remove_non_latin_lines(lines)

    lines = _remove_garbage_lines(lines)

    lines = _remove_single_char_junk(lines)

    lines = _collapse_blank_lines(lines)

    text = "\n".join(lines)

    text = _join_linebreak_hyphens(text)
    text = _join_caps_hyphens(text)
    text = _join_dictionary_hyphens(text)

    text = _apply_ocr_subs(text)
    text = _normalize_roman_tokens(text)
    text = _fix_missing_spaces(text)

    text = _correct_ocr_words(text)

    text = _split_merged_words(text)
    text = _apply_contextual_legal_fixes(text)

    text = _fix_missing_spaces(text)

    text = _normalize_whitespace(text)

    return text

def process_file(input_path, output_path) -> dict:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Refusing to overwrite input file. Use a different output path."
        )

    if input_path.suffix.lower() == ".json":
        blocks, raw_text = extract_text_from_json(input_path)

        raw_txt_text = None
        raw_candidates = [
            input_path.with_suffix(".txt"),
            input_path.parent / "Raw_ocr.txt",
            input_path.parent / "raw_ocr.txt",
        ]
        for raw_txt_path in raw_candidates:
            if raw_txt_path.exists():
                with open(raw_txt_path, "r", encoding="utf-8") as f:
                    raw_txt_text = f.read()
                break
    else:
        blocks       = None
        raw_txt_text = None
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

    cleaned = normalize_page_text(raw_text, blocks=blocks)

    cleaned = validate_final_output(
        cleaned,
        blocks=blocks,
        raw_txt_text=raw_txt_text,
    )
    cleaned = _apply_contextual_legal_fixes(cleaned)
    cleaned = _normalize_whitespace(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    in_chars  = len(raw_text)
    out_chars = len(cleaned)
    reduction = (
        round(100.0 * (1.0 - out_chars / in_chars), 1) if in_chars else 0.0
    )

    return {
        "input_file":    str(input_path),
        "output_path":   str(output_path),
        "input_chars":   in_chars,
        "output_chars":  out_chars,
        "reduction_pct": reduction,
    }


def process_directory(input_dir, output_dir, pattern: str = "*.json") -> list:

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"  No files matching '{pattern}' in {input_dir}")
        return []

    results = []
    for f in files:
        out_path = output_dir / f.name
        try:
            stats = process_file(f, out_path)
            results.append(stats)
            print(
                f"  OK  {f.name:35s}  "
                f"{stats['input_chars']:>7,} -> {stats['output_chars']:>7,} chars  "
                f"({stats['reduction_pct']:>5.1f}% reduction)"
            )
        except Exception as exc:
            print(f"  ERR {f.name}: {exc}")
            results.append({"input_file": str(f.name), "error": str(exc)})

    return results



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process PaddleOCR-VL output files for Spanish legal documents.\n"
            "Accepts a single .json/.txt file or a directory of files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python post_processing.py --input result_1.json\n"
            "  python post_processing.py --input result_1.json --output cleaned.txt\n"
            "  python post_processing.py --input ./output_pdf/ --output ./cleaned/\n"
        ),
    )
    parser.add_argument("--input",   "-i", required=True,
                        help="Input .json/.txt file OR directory")
    parser.add_argument("--output",  "-o", default=None,
                        help="Output file or directory (auto-named if omitted)")
    parser.add_argument("--pattern", default="*.json",
                        help="Glob pattern for directory mode (default: *.json)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        output_dir = (
            Path(args.output)
            if args.output
            else input_path.parent / (input_path.name + "_clean")
        )
        print(f"\nInput  : {input_path}")
        print(f"Output : {output_dir}\n")
        stats_list = process_directory(input_path, output_dir, args.pattern)
        total_in  = sum(s.get("input_chars",  0) for s in stats_list)
        total_out = sum(s.get("output_chars", 0) for s in stats_list)
        pct = round(100 * (1 - total_out / total_in), 1) if total_in else 0
        print(f"\n{'=' * 60}")
        print(f"  Files processed : {len(stats_list)}")
        print(f"  Total input     : {total_in:,} chars")
        print(f"  Total output    : {total_out:,} chars")
        print(f"  Noise removed   : {pct}%")

    elif input_path.is_file():
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_name(
                f"{input_path.stem}_clean{input_path.suffix}"
            )
        )
        print(f"\nProcessing : {input_path.name}")
        stats = process_file(input_path, output_path)
        print(f"  Input chars  : {stats['input_chars']:,}")
        print(f"  Output chars : {stats['output_chars']:,}")
        print(f"  Noise removed: {stats['reduction_pct']}%")
        print(f"  Saved to     : {stats['output_path']}")

    else:
        print(f'Error: "{input_path}" is not a valid file or directory.')
        raise SystemExit(1)


if __name__ == "__main__":
    main()
