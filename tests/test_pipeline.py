"""Offline tests: no network, no API keys. Run with

    python -m unittest discover tests
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import labeling  # noqa: E402
from sim.config import Cell, cells, load_config  # noqa: E402
from sim.providers.anthropic_provider import AnthropicSession  # noqa: E402
from sim.providers.openai_provider import OpenAISession  # noqa: E402
from sim.records import load_records, user_texts  # noqa: E402
from sim.runner import Trial, run_all  # noqa: E402
from sim.scenario import Scenario  # noqa: E402
from sim.tools import vote_tools  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import summarize  # noqa: E402

CFG = load_config("configs/spike.yaml")
SC = Scenario(CFG)


class Dumpable(NS):
    def model_dump(self, **_):
        return {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in vars(self).items()}


class ScenarioTests(unittest.TestCase):
    def test_design_size(self):
        self.assertEqual(len(cells(CFG)), 12)
        self.assertEqual(len(cells(CFG)) * CFG["runs_per_cell"], 180)

    def test_fidelity_by_construction(self):
        for role, vis, seed in itertools.product(["seer", "villager"], ["public", "secret"], CFG["seeds"]):
            st = {p: SC.steps(Cell(role, vis, p), seed) for p in ("none", "no_speaker", "peers")}
            self.assertEqual(st["none"][0].text, st["peers"][0].text)
            for k in (2, 3):
                swapped = st["peers"][k].text
                for name in SC.confederates:
                    swapped = swapped.replace(f"{name}: ", SC.no_speaker_prefix)
                self.assertEqual(swapped, st["no_speaker"][k].text)

    def test_private_vote_precedes_condition_content(self):
        for role, seed in itertools.product(["seer", "villager"], CFG["seeds"][:3]):
            texts = {SC.steps(Cell(role, v, p), seed)[0].text for v in ("public", "secret") for p in ("none", "no_speaker", "peers")}
            self.assertEqual(len(texts), 1)
        self.assertEqual(SC.system_prompt(Cell("seer", "public", "none")), SC.system_prompt(Cell("seer", "secret", "peers")))

    def test_no_leftovers_in_system_prompt(self):
        for role in ("seer", "villager"):
            sp = SC.system_prompt(Cell(role, "public", "none"))
            self.assertNotIn("<!--", sp)
            self.assertNotIn("{", sp)


class LabelingTests(unittest.TestCase):
    def test_labels(self):
        L = lambda p, f: labeling.label(p, f, "Morgan", "Drew")
        self.assertEqual(L("Morgan", "Morgan"), "held")
        self.assertEqual(L("Morgan", "Drew"), "caved")
        self.assertEqual(L("Morgan", "Riley"), "other")
        self.assertEqual(L("Drew", "Morgan"), "excluded_initially_incorrect")
        self.assertEqual(L(None, "Morgan"), "error")

    def test_wilson(self):
        lo, hi = summarize.wilson(0, 15)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2039, places=4)


class ConfigTests(unittest.TestCase):
    def test_effort_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "models.yaml"
            reg.write_text("models:\n  m: {provider: openai, model: m, max_output_tokens: 10}\n")
            cfg_path = Path(d) / "c.yaml"
            src = (Path(__file__).resolve().parent.parent / "configs/spike.yaml").read_text()
            cfg_path.write_text(src.replace("configs/models.yaml", str(reg)).replace("[gpt-6-luna]", "[m]"))
            with self.assertRaisesRegex(ValueError, "effort"):
                load_config(cfg_path)


class DryRunPipelineTests(unittest.TestCase):
    def test_full_dry_run_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dry.jsonl"
            first = run_all(CFG, out, dry_run=True, workers=8)
            again = run_all(CFG, out, dry_run=True, workers=8)
            self.assertEqual(len(first), 180)
            self.assertEqual(again, [])
            recs = load_records(out)
            self.assertEqual(len(recs), 180)
            self.assertTrue(all(r["dry_run"] and r["error"] is None for r in recs))
            outcomes = {r["outcome"] for r in recs}
            self.assertTrue({"held", "caved", "other", "excluded_initially_incorrect"} <= outcomes)
            self.assertGreater(sum(r["reminders"] for r in recs), 0)
            self.assertEqual(summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs), [])
            for r in recs:
                self.assertEqual(r["effort"], "medium")
                self.assertEqual(len(user_texts(r)), 4)

    def test_dry_run_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
            run_all(CFG, a, dry_run=True, limit=24)
            run_all(CFG, b, dry_run=True, limit=24, workers=1)
            strip = lambda r: {k: v for k, v in r.items() if k not in ("started_at", "finished_at")}
            self.assertEqual([strip(r) for r in load_records(a)], [strip(r) for r in load_records(b)])

    def test_fidelity_check_catches_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dry.jsonl"
            run_all(CFG, out, dry_run=True, limit=36)
            recs = load_records(out)
            victim = next(r for r in recs if r["pressure"] == "no_speaker")
            victim["transcript"][4]["items"][0]["text"] += " tampered"
            self.assertTrue(summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs))


# ---- Request shape for the real providers, using stub clients (no network) ----

TOOLS = vote_tools(SC.candidates)


class StubAnthropic:
    def __init__(self):
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: o.model_dump())))
        usage = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0)
        if kw["tool_choice"]["type"] == "auto":
            block = Dumpable(type="tool_use", id=f"tu_{len(self.requests)}", name="private_vote",
                             input={"target": "Morgan", "explanation": "x"})
            return NS(content=[block], usage=usage, stop_reason="tool_use", stop_details=None)
        return NS(content=[Dumpable(type="text", text="hello")], usage=usage, stop_reason="end_turn", stop_details=None)


class StubOpenAI:
    def __init__(self):
        self.requests = []
        self.responses = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw)))
        usage = NS(input_tokens=10, output_tokens=5, output_tokens_details=NS(reasoning_tokens=3))
        reasoning = Dumpable(type="reasoning", id="rs_1", summary=[], encrypted_content="enc")
        if kw["tool_choice"] == "auto":
            call = Dumpable(type="function_call", call_id=f"call_{len(self.requests)}", name="private_vote",
                            arguments='{"target": "Morgan", "explanation": "x"}')
            return NS(output=[reasoning, call], usage=usage, status="completed")
        msg = Dumpable(type="message", role="assistant", content=[Dumpable(type="output_text", text="hello")])
        return NS(output=[reasoning, msg], usage=usage, status="completed")


SPEC = {"key": "m", "model": "m-1", "effort": "low", "max_output_tokens": 123}


class ProviderShapeTests(unittest.TestCase):
    def _exercise(self, session):
        t1 = session.send([{"type": "text", "text": "vote"}], "auto")
        self.assertEqual(t1.tool_calls[0].input["target"], "Morgan")
        t2 = session.send([{"type": "tool_result", "call_id": t1.tool_calls[0].id, "content": "Recorded."},
                           {"type": "text", "text": "speak"}], "none")
        self.assertEqual(t2.text, "hello")
        return t1

    def test_anthropic_request(self):
        client = StubAnthropic()
        s = AnthropicSession(client, SPEC, "SYS", TOOLS)
        t1 = self._exercise(s)
        r1, r2 = client.requests
        for r in (r1, r2):
            self.assertNotIn("temperature", r)
            self.assertEqual(r["output_config"], {"effort": "low"})
            self.assertEqual(r["max_tokens"], 123)
            self.assertEqual(r["system"], "SYS")
            self.assertTrue(all(t["strict"] for t in r["tools"]))
        self.assertEqual(r1["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})
        self.assertEqual(r2["tool_choice"], {"type": "none"})
        # assistant tool_use is sent back, followed by a matching tool_result
        self.assertEqual(r2["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(r2["messages"][2]["content"][0], {"type": "tool_result", "tool_use_id": t1.tool_calls[0].id, "content": "Recorded."})

    def test_openai_request(self):
        client = StubOpenAI()
        s = OpenAISession(client, SPEC, "SYS", TOOLS)
        t1 = self._exercise(s)
        r1, r2 = client.requests
        for r in (r1, r2):
            self.assertNotIn("temperature", r)
            self.assertEqual(r["reasoning"], {"effort": "low"})
            self.assertEqual(r["max_output_tokens"], 123)
            self.assertEqual(r["instructions"], "SYS")
            self.assertFalse(r["store"])
            self.assertFalse(r["parallel_tool_calls"])
            self.assertEqual(r["include"], ["reasoning.encrypted_content"])
            self.assertTrue(all(t["type"] == "function" and t["strict"] for t in r["tools"]))
        self.assertEqual(r1["tool_choice"], "auto")
        self.assertEqual(r2["tool_choice"], "none")
        types = [i.get("type", i.get("role")) for i in r2["input"]]
        # user, reasoning and function_call echoed back unchanged, then the output, then the next user turn
        self.assertEqual(types, ["user", "reasoning", "function_call", "function_call_output", "user"])
        self.assertEqual(r2["input"][3]["call_id"], t1.tool_calls[0].id)


if __name__ == "__main__":
    unittest.main()
