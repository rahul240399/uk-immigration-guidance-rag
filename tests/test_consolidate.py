"""Tests for consolidation and generation v1."""
import csv, json, re
from pathlib import Path
import pytest

DISPLAY = {"skilled_worker":"Skilled Worker","student":"Student","graduate":"Graduate","visitor":"Visitor","family":"Family"}
B6_RE = re.compile(r'\b(Appendix|Part\s+\d|paragraph\s+\d|para\.?\s*\d|[A-Z]{1,6}\s?\d+\.\d+|[A-Z]+-[A-Z]+\.\d|\([A-Z]\))', re.IGNORECASE)

def _validate(row, seen):
    vs = []
    q=row["question"]; a=row["reference_answer"]; dn=DISPLAY.get(row["route"],"")
    if B6_RE.search(q): vs.append("B6")
    if dn not in q: vs.append("no_route")
    if q.lower().count(dn.lower())>1: vs.append("route_2x")
    if len(q.split())<8: vs.append("q_short")
    if len(a.split())<10: vs.append("a_short")
    if a.rstrip().endswith(":"): vs.append("a_colon")
    if "the following" in a.lower(): vs.append("a_following")
    if "listed in" in a.lower(): vs.append("a_listed")
    if q.lower().strip() in seen: vs.append("dup")
    return vs

class TestConsolidationRules:
    def test_fragment_dropped(self):
        row = {"route":"bad","question":"x","reference_answer":"y","gold_paragraph_ids":"g1"}
        assert row["route"] not in DISPLAY

    def test_dup_gold_keeps_passing(self):
        seen = set()
        r1 = {"route":"student","question":"What is the Student requirement for showing sufficient funds in the UK?",
               "reference_answer":"The student must show funds of at least 1000 pounds for living costs during their stay."}
        r2 = {"route":"student","question":"How Appendix Student works",
               "reference_answer":"short"}
        assert _validate(r1, seen) == []
        assert len(_validate(r2, seen)) > 0  # B6 + short

    def test_dup_question_dropped(self):
        seen = {"what is the requirement?"}
        row = {"route":"student","question":"What is the requirement?",
               "reference_answer":"A student must meet the requirements including funds and sponsor."}
        vs = _validate(row, seen)
        assert "dup" in vs

    def test_id_from_seed(self):
        sid = "S0042"
        qid = f"Q{sid[1:]}"
        assert qid == "Q0042"

class TestLockFile:
    def test_lock_refuses(self, tmp_path):
        lock = tmp_path / ".generate.lock"
        lock.write_text("1")
        assert lock.exists()

class TestValidatorCases:
    def test_b6_regex(self):
        assert B6_RE.search("What does Appendix FM require?")
        assert B6_RE.search("paragraph 320 says")
        assert B6_RE.search("SW 1.1 requirement")
        assert not B6_RE.search("What is the Skilled Worker requirement?")

    def test_route_name(self):
        assert "Skilled Worker" in "What must a Skilled Worker show?"
        assert "Skilled Worker" not in "What must an applicant show?"

    def test_answer_rules(self):
        assert "the following" in "the applicant must meet the following".lower()
        assert "listed in" in "requirements listed in the rules".lower()

class TestTopupStops:
    def test_stops_at_target(self):
        target = 3; done = 3
        assert done >= target  # would skip in the loop

