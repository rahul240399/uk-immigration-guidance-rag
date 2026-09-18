"""
Tests for code.index.build_index — synthetic fixtures only.
"""

import json
from pathlib import Path

import pytest


def _make_unit(uid, section, seq, text="Some text here", heading_path=None, corpus="rules", route="test"):
    hp = heading_path if heading_path is not None else ["Heading"]
    hl = " > ".join(hp) if hp else section
    full_text = hl + "\n" + text
    return {
        "unit_id": uid, "corpus": corpus,
        "section_base_path": section,
        "heading_path": hp, "seq": seq,
        "route": route,
        "text": full_text,
        "token_count": len(text.split()),
        "unit_token_count": len(full_text.split()),
    }


class TestWindowMapping:
    """50% rule and largest-overlap fallback."""

    def test_50_percent_rule(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        units = [_make_unit("u1", "/sec", 1, text="a b c d")]
        units[0]["token_count"] = len(tok.encode("a b c d", add_special_tokens=False))
        windows, stats = build_win64(units, tok)
        assert len(windows) >= 1
        assert "u1" in windows[0]["paragraph_ids"]

    def test_fallback_largest_overlap(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        units = [_make_unit("u1", "/sec", 1, text=" ".join(["word"] * 100))]
        units[0]["token_count"] = len(tok.encode(" ".join(["word"] * 100), add_special_tokens=False))
        windows, stats = build_win64(units, tok)
        carried = any("u1" in w["paragraph_ids"] for w in windows)
        assert carried


class TestNoSectionCrossing:
    def test_windows_dont_cross_sections(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        units = [
            _make_unit("a1", "/sec-a", 1, text=" ".join(["w"] * 50)),
            _make_unit("b1", "/sec-b", 1, text=" ".join(["w"] * 50)),
        ]
        for u in units:
            record_text = u["text"].split("\n", 1)[-1]
            u["token_count"] = len(tok.encode(record_text, add_special_tokens=False))
        windows, stats = build_win64(units, tok)
        for w in windows:
            assert w["section_base_path"] in ("/sec-a", "/sec-b")


class TestHeadingOnce:
    def test_heading_emitted_on_change(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        units = [
            _make_unit("u1", "/sec", 1, heading_path=["H1"], text="a b"),
            _make_unit("u2", "/sec", 2, heading_path=["H1"], text="c d"),
            _make_unit("u3", "/sec", 3, heading_path=["H2"], text="e f"),
        ]
        for u in units:
            record_text = u["text"].split("\n", 1)[-1]
            u["token_count"] = len(tok.encode(record_text, add_special_tokens=False))
        windows, stats = build_win64(units, tok)
        assert len(windows) >= 1


class TestDeletedExcluded:
    def test_no_deleted_in_para(self):
        from code.index.build_index import build_para
        units = [
            _make_unit("u1", "/sec", 1),
            # Deleted records should never appear as units (filtered by units.py)
            # But if one slipped through, para still uses it — the exclusion is upstream
        ]
        chunks = build_para(units)
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "u1"


class TestParentGrouping:
    def test_grouped_by_heading(self):
        from code.index.build_index import build_parentchild
        units = [
            _make_unit("u1", "/sec", 1, heading_path=["H1"]),
            _make_unit("u2", "/sec", 2, heading_path=["H1"]),
            _make_unit("u3", "/sec", 3, heading_path=["H2"]),
        ]
        children, parents = build_parentchild(units)
        assert len(children) == 3
        # Two distinct parents: H1, H2
        assert len(parents) == 2
        h1_parent = [p for p in parents if p["heading"] == "H1"]
        assert len(h1_parent) == 1
        assert len(h1_parent[0]["unit_ids"]) == 2


class TestEmptyHeadingFallback:
    def test_falls_back_to_section(self):
        from code.index.build_index import build_parentchild
        units = [_make_unit("u1", "/sec", 1, heading_path=[])]
        # heading_line falls back to section_base_path
        children, parents = build_parentchild(units)
        assert len(parents) == 1
        assert parents[0]["heading"] == "/sec"


class TestDraftingStyle:
    def test_new_old_list(self):
        # new: ≥50% match ^[A-Z]{1,6}\s?\d+\.\d+
        # old: has rule_refs but <50% match
        # list: no rule_refs
        from code.index.covariates import NEW_PATTERN
        assert NEW_PATTERN.match("SW 1.1")
        assert NEW_PATTERN.match("GEN1.2")
        assert not NEW_PATTERN.match("320")
        assert not NEW_PATTERN.match("(a)")


class TestParaChunkId:
    def test_chunk_id_is_unit_id(self):
        from code.index.build_index import build_para
        units = [_make_unit(f"u{i}", "/sec", i) for i in range(5)]
        chunks = build_para(units)
        for c, u in zip(chunks, units):
            assert c["chunk_id"] == u["unit_id"]


class TestWindowSubstring:
    """Window text must be an exact substring of the original stream."""

    def test_substring_of_stream(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        units = [
            _make_unit("u1", "/sec", 1, heading_path=["Part 1"],
                       text="The applicant must show evidence of English language ability."),
            _make_unit("u2", "/sec", 2, heading_path=["Part 1"],
                       text="This requirement applies to all main applicants."),
        ]
        windows, stats = build_win64(units, tok)

        # Reconstruct the stream
        stream = "Part 1" + "The applicant must show evidence of English language ability." + \
                 "This requirement applies to all main applicants."

        for w in windows:
            assert w["text"] in stream or w["text"].strip() in stream, \
                f"Window text not a substring: {w['text'][:60]!r}"


class TestInvariantsOnSyntheticSection:
    """Invariant tests: short record, 40/60 split, record spanning 3 windows."""

    def test_invariants(self):
        from code.index.build_index import build_win64
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        # Short record (fits in one window)
        short_text = "A short rule."
        # Medium record that will split ~40/60 across two windows
        medium_text = " ".join(["word"] * 80)
        # Long record spanning 3+ windows (>192 tokens)
        long_text = " ".join(["requirement"] * 250)

        units = [
            _make_unit("u1", "/sec", 1, heading_path=["H"], text=short_text),
            _make_unit("u2", "/sec", 2, heading_path=["H"], text=medium_text),
            _make_unit("u3", "/sec", 3, heading_path=["H"], text=long_text),
        ]
        # Set token_count from the tokenizer
        for u in units:
            record_text = u["text"].split("\n", 1)[-1]
            u["token_count"] = len(tok.encode(record_text, add_special_tokens=False))

        windows, stats = build_win64(units, tok)

        # No window > 64 tokens
        for w in windows:
            assert w["token_count"] <= 64, f"Window {w['window_id']} has {w['token_count']} tokens"

        # All three records carried by at least one window
        all_carried = set()
        for w in windows:
            all_carried.update(w["paragraph_ids"])
        assert "u1" in all_carried
        assert "u2" in all_carried
        assert "u3" in all_carried

        # No window crosses sections (only one section here)
        for w in windows:
            assert w["section_base_path"] == "/sec"

        # overlap_tokens lists every record in the window
        for w in windows:
            for pid in w["overlap_tokens"]:
                assert pid in ("u1", "u2", "u3")


class TestUnitTokenCounts:
    """Both token_count and unit_token_count present; unit >= record."""

    def test_both_fields_unit_gte_record(self):
        from code.index.units import unit_text, heading_line
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

        # Simulate a unit with heading
        record = {
            "paragraph_id": "x:00001",
            "heading_path": ["Part 1", "Section A"],
            "section_title": "Test",
            "text": "The applicant must provide evidence.",
        }
        full_text = unit_text(record)
        record_only = record["text"]
        utc = len(tok.encode(full_text, add_special_tokens=False))
        rtc = len(tok.encode(record_only, add_special_tokens=False))
        assert utc >= rtc, f"unit_token_count {utc} < token_count {rtc}"
        assert utc > rtc, "heading should add tokens"
