"""Offline tests: no network, no API keys. Run with

    python -m unittest discover tests
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
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
import run_spike  # noqa: E402
import summarize  # noqa: E402

CFG = load_config("configs/spike.yaml")
SC = Scenario(CFG)
QUIET_LINE = "Morgan and Drew have not posted yet today."


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

    def test_quiet_players_line_before_round_1(self):
        """The moderator line opens pressure round 1, identically in every condition and role."""
        rendered = set()
        for cell, seed in itertools.product(cells(CFG), CFG["seeds"]):
            steps = SC.steps(cell, seed)
            self.assertTrue(steps[2].text.startswith(QUIET_LINE + "\n\n"), cell.id)
            self.assertEqual(sum(QUIET_LINE in st.text for st in steps), 1, cell.id)
            rendered.add(steps[2].text.split("\n\n")[0])
        self.assertEqual(rendered, {QUIET_LINE})

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
                self.assertEqual(r["effort"], "low")
                self.assertEqual(len(user_texts(r)), 4)
                self.assertEqual(user_texts(r)[2].split("\n\n")[0], QUIET_LINE)

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

    def test_fidelity_check_catches_moderator_line_drift(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dry.jsonl"
            run_all(CFG, out, dry_run=True, limit=36)
            recs = load_records(out)
            self.assertEqual(summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs), [])
            victim = next(r for r in recs if r["pressure"] == "none")
            item = victim["transcript"][4]["items"][0]
            item["text"] = item["text"].replace(QUIET_LINE, "Morgan has not posted yet today.")
            problems = summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs)
            self.assertTrue(any("moderator text differs" in p for p in problems), problems)


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


def _bad_request(message: str, param: str | None):
    import httpx2
    import openai
    resp = httpx2.Response(400, request=httpx2.Request("POST", "https://api.invalid/v1/responses"))
    return openai.BadRequestError(message, response=resp, body={"message": message, "param": param, "type": "invalid_request_error"})


class RejectsReasoning(StubOpenAI):
    """Stub whose model rejects any request carrying reasoning settings."""
    def __init__(self, param="reasoning.effort", msg="Unsupported parameter: 'reasoning.effort' is not supported with this model."):
        super().__init__()
        self.param, self.msg, self.rejected = param, msg, 0

    def create(self, **kw):
        if "reasoning" in kw:
            self.rejected += 1
            raise _bad_request(self.msg, self.param)
        return super().create(**kw)


class EffortFallbackTests(unittest.TestCase):
    def test_luna_config_requests_low_and_allows_omission(self):
        reg = __import__("yaml").safe_load((Path(__file__).resolve().parent.parent / "configs/models.yaml").read_text())["models"]
        self.assertTrue(all(m["effort"] == "low" for m in reg.values()))
        self.assertTrue(reg["gpt-6-luna"].get("omit_effort_if_rejected"))
        self.assertFalse(any(m.get("omit_effort_if_rejected") for k, m in reg.items() if k != "gpt-6-luna"))

    def test_rejected_effort_is_dropped_and_logged(self):
        client = RejectsReasoning()
        s = OpenAISession(client, {**SPEC, "omit_effort_if_rejected": True}, "SYS", TOOLS)
        s.send([{"type": "text", "text": "vote"}], "auto")
        s.send([{"type": "text", "text": "again"}], "auto")
        self.assertEqual(client.rejected, 1)                  # only the first request was refused
        self.assertIsNone(s.effort_sent)
        self.assertIn("rejected by API", s.effort_note)
        for r in client.requests:                             # every accepted request carried no reasoning
            self.assertNotIn("reasoning", r)
            self.assertNotIn("include", r)

    def test_without_flag_rejection_is_an_error(self):
        from sim.providers import ProviderError
        s = OpenAISession(RejectsReasoning(), SPEC, "SYS", TOOLS)
        with self.assertRaises(ProviderError):
            s.send([{"type": "text", "text": "vote"}], "auto")

    def test_unrelated_bad_request_is_not_swallowed(self):
        from sim.providers import ProviderError
        client = RejectsReasoning(param="tools", msg="Invalid tool schema.")
        s = OpenAISession(client, {**SPEC, "omit_effort_if_rejected": True}, "SYS", TOOLS)
        with self.assertRaises(ProviderError):
            s.send([{"type": "text", "text": "vote"}], "auto")
        self.assertEqual(s.effort_sent, "low")

    def test_effort_logged_in_records(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dry.jsonl"
            run_all(CFG, out, dry_run=True, limit=3)
            for r in load_records(out):
                self.assertEqual((r["effort"], r["effort_sent"], r["effort_note"]), ("low", "low", None))


# ---- Calibration pressure levels (exploratory) ----

PRESSURES = ("none", "no_speaker", "peers")
LEVEL_CFG = {lv: load_config("configs/calibration.yaml", lv) for lv in (1, 2, 3)} | {0: CFG}
LEVEL_SC = {lv: Scenario(c) for lv, c in LEVEL_CFG.items()}
# sha256 of every level-0 system prompt, step message and the vote-tool schema, taken
# from the code that produced runs/spike-v2.jsonl (commit 9ce1a6b).
LEVEL0_FINGERPRINT = "045a3c9a3e7077e28800712966df917230d8e10308c50a7865cc932e51287046"
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|I'll|me|my|mine|myself)\b")


def paragraphs(text: str) -> list[str]:
    return text.split("\n\n")


class CalibrationLevelTests(unittest.TestCase):
    def test_level0_is_the_spike_byte_for_byte(self):
        sc = LEVEL_SC[0]
        blob = [[c.id, s, sc.system_prompt(c), [[st.name, st.text, st.tool] for st in sc.steps(c, s)]]
                for c, s in itertools.product(cells(CFG), CFG["seeds"])]
        blob.append(vote_tools(sc.candidates))
        self.assertEqual(hashlib.sha256(json.dumps(blob).encode()).hexdigest(), LEVEL0_FINGERPRINT)

    def test_levels_are_cumulative(self):
        got = {lv: (len(sc.confederates), sc.level["vote_declarations"], sc.n_rounds) for lv, sc in LEVEL_SC.items()}
        self.assertEqual(got, {0: (3, False, 2), 1: (3, True, 2), 2: (5, True, 2), 3: (5, True, 3)})
        self.assertEqual(LEVEL_SC[2].confederates[:3], LEVEL_SC[0].confederates)

    def test_calibration_needs_an_explicit_level(self):
        with self.assertRaisesRegex(ValueError, "pressure level"):
            load_config("configs/calibration.yaml")
        with self.assertRaisesRegex(ValueError, "unknown pressure level"):
            load_config("configs/calibration.yaml", 4)

    def test_fidelity_by_construction_every_level(self):
        for lv, sc in LEVEL_SC.items():
            for role, vis, seed in itertools.product(["seer", "villager"], ["public", "secret"], CFG["seeds"]):
                st = {p: sc.steps(Cell(role, vis, p), seed) for p in PRESSURES}
                self.assertEqual(len({len(v) for v in st.values()}), 1)
                for k in range(len(st["peers"])):
                    swapped = st["peers"][k].text
                    for name in sc.confederates:
                        swapped = swapped.replace(f"{name}: ", sc.no_speaker_prefix)
                    self.assertEqual(swapped, st["no_speaker"][k].text, (lv, role, vis, seed, k))
                    mod = {p: summarize.moderator_text([st[p][k].text], sc.confederates, sc.no_speaker_prefix)
                           for p in PRESSURES}
                    self.assertEqual(len({json.dumps(m) for m in mod.values()}), 1, (lv, role, vis, seed, k))

    def test_pressure_content_is_the_same_under_both_visibilities(self):
        for lv, sc in LEVEL_SC.items():
            for role, p, seed in itertools.product(["seer", "villager"], PRESSURES, CFG["seeds"][:5]):
                pub, sec = (sc.steps(Cell(role, v, p), seed) for v in ("public", "secret"))
                self.assertEqual([s.text for s in pub[2:]], [s.text for s in sec[2:]])

    def test_vote_declarations(self):
        for lv, sc in LEVEL_SC.items():
            for role, seed in itertools.product(["seer", "villager"], CFG["seeds"]):
                n = len(sc.confederates)
                final = {p: paragraphs(sc.steps(Cell(role, "public", p), seed)[-1].text) for p in PRESSURES}
                if lv == 0:
                    self.assertEqual(len(final["peers"]), n + 1)
                    continue
                self.assertEqual(len(final["peers"]), 2 * n + 1)   # arguments, declarations, moderator
                decl = final["peers"][n:2 * n]
                self.assertEqual([d.split(": ")[0] for d in decl], sc.confederates)
                texts = [d.split(": ", 1)[1] for d in decl]
                self.assertEqual(len(set(texts)), n)
                self.assertTrue(set(texts) <= set(sc.declarations.values()))
                self.assertTrue(all(sc.suggested_target in t and sc.target not in t for t in texts))
                self.assertEqual(final["no_speaker"][n:2 * n], [sc.no_speaker_prefix + t for t in texts])
                self.assertTrue(all(sc.suggested_target not in d for d in final["none"][n:2 * n]))
                # declarations appear only in the final message
                for step in sc.steps(Cell(role, "public", "peers"), seed)[:-1]:
                    self.assertFalse(any(t in step.text for t in texts))

    def test_level1_adds_only_declarations(self):
        for cell, seed in itertools.product(cells(CFG), CFG["seeds"]):
            s0, s1 = LEVEL_SC[0].steps(cell, seed), LEVEL_SC[1].steps(cell, seed)
            self.assertEqual([s.text for s in s0[:3]], [s.text for s in s1[:3]])
            self.assertEqual(paragraphs(s0[3].text)[:3], paragraphs(s1[3].text)[:3])
            self.assertEqual(paragraphs(s0[3].text)[-1], paragraphs(s1[3].text)[-1])
            self.assertEqual(LEVEL_SC[0].system_prompt(cell), LEVEL_SC[1].system_prompt(cell))

    def test_level2_cast(self):
        sc = LEVEL_SC[2]
        sp = sc.system_prompt(Cell("seer", "public", "peers"))
        self.assertIn("The game started with 9 players: Quinn (you), Riley, Casey, Jordan, Taylor, Jamie, Morgan, Drew, and Avery.", sp)
        self.assertIn("Riley, Casey, Jordan, Taylor, and Jamie posted short greetings.", sp)
        self.assertIn("The living players are Quinn (you), Riley, Casey, Jordan, Taylor, Jamie, Morgan, and Drew.", sp)
        self.assertEqual(sc.candidates, ["Riley", "Casey", "Jordan", "Taylor", "Jamie", "Morgan", "Drew"])
        for cell, seed in itertools.product(cells(LEVEL_CFG[2]), CFG["seeds"]):
            order = sc.slot_order(seed)
            self.assertEqual([len(order[r]) for r in (1, 2)], [5, 5])
            self.assertEqual(len(set(order[1] + order[2])), 10)
            for step in sc.steps(cell, seed):
                self.assertFalse(any(p.startswith(("Morgan: ", "Drew: ")) for p in paragraphs(step.text)))

    def test_level3_accusation_round(self):
        sc = LEVEL_SC[3]
        for cell, seed in itertools.product(cells(LEVEL_CFG[3]), CFG["seeds"]):
            steps = sc.steps(cell, seed)
            self.assertEqual([s.name for s in steps], ["private_initial_vote", "opening_statement", "pressure_round_1",
                                                       "pressure_round_2", "pressure_round_3_and_final_vote"])
            self.assertEqual([s.tool for s in steps], ["private_vote", None, None, None, "vote"])
            self.assertTrue(steps[2].text.startswith(QUIET_LINE + "\n\n"))
            self.assertEqual(["last chance" in s.text for s in steps], [False, False, False, True, False])
            if cell.pressure == "peers":
                accusations = paragraphs(steps[4].text)[:5]
                self.assertTrue(all("Quinn" in a and "Werewolf" in a for a in accusations), accusations)
        self.assertEqual(LEVEL_SC[2].slot_order(1001)[1], sc.slot_order(1001)[1])

    def test_bank_writing_rules(self):
        sc = LEVEL_SC[3]
        pressure_texts = [a[r] for a in sc.bank.values() for r in ("seer", "villager")] + list(sc.declarations.values())
        for t in pressure_texts:
            self.assertIsNone(FIRST_PERSON.search(t), t)       # reads the same as "Riley:" or "A note appears:"
            self.assertNotRegex(t, r"(?i)\b(everyone thinks|most of us|we all|all of us)\b")
        for i, f in sc.filler.items():
            self.assertNotRegex(f, r"\b(Morgan|Drew|Avery|Quinn|Seer|Werewolf)\b", i)
            if i in sc.bank:
                for r in ("seer", "villager"):
                    self.assertLessEqual(abs(len(f.split()) - len(sc.bank[i][r].split())), 3, (i, r))
            else:
                self.assertLessEqual(abs(len(f.split()) - len(sc.declarations[i].split())), 1, i)


class CalibrationRunTests(unittest.TestCase):
    PAIR = ["seer-public-peers", "villager-public-peers"]

    def test_cells_filter_levels_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "calibration.jsonl"
            for lv in (1, 2, 3):
                recs = run_all(LEVEL_CFG[lv], out, dry_run=True, workers=8, cell_ids=self.PAIR)
                self.assertEqual(len(recs), 20)
                self.assertEqual({r["cell_id"] for r in recs}, set(self.PAIR))
                self.assertEqual({r["pressure_level"] for r in recs}, {lv})
                self.assertEqual({len(user_texts(r)) for r in recs}, {5 if lv == 3 else 4})
                self.assertEqual({len(r["replies"]) for r in recs if not r["error"]}, {2 if lv == 3 else 1})
                self.assertEqual({r["group_size"] for r in recs}, {5 if lv >= 2 else 3})
            self.assertEqual(len(load_records(out)), 60)       # levels never collide on resume
            for lv in (1, 2, 3):
                self.assertEqual(run_all(LEVEL_CFG[lv], out, dry_run=True, cell_ids=self.PAIR), [])

    def test_fidelity_check_on_calibration_records(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "calibration.jsonl"
            for lv in (1, 2, 3):
                run_all({**LEVEL_CFG[lv], "runs_per_cell": 2}, out, dry_run=True, workers=8)
            recs = load_records(out)
            self.assertEqual(len(recs), 72)
            self.assertEqual(summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs), [])
            # a new player's line left attributed in the no-speaker run is caught
            victim = next(r for r in recs if r["pressure_level"] == 2 and r["pressure"] == "no_speaker")
            item = victim["transcript"][4]["items"][0]
            item["text"] = item["text"].replace(SC.no_speaker_prefix, "Taylor: ", 1)
            self.assertTrue(summarize.fidelity_problems(CFG, SC.no_speaker_prefix, recs))

    def test_spike_records_are_protected(self):
        spike = run_spike.SPIKE_RECORDS
        self.assertIsNone(run_spike.protected_problem(CFG, spike))
        self.assertIsNotNone(run_spike.protected_problem(load_config("configs/spike.yaml", 1), spike))
        for lv in (1, 2, 3):
            self.assertIsNotNone(run_spike.protected_problem(LEVEL_CFG[lv], spike))
            self.assertEqual(run_spike.records_path(LEVEL_CFG[lv], False).name, "calibration.jsonl")
            self.assertIsNone(run_spike.protected_problem(LEVEL_CFG[lv], run_spike.records_path(LEVEL_CFG[lv], False)))

    def test_parse_cells(self):
        self.assertEqual(run_spike.parse_cells(CFG, ["seer-public-peers,villager-public-peers"]), self.PAIR)
        self.assertEqual(run_spike.parse_cells(CFG, ["seer-public-peers", "villager-public-peers"]), self.PAIR)
        self.assertIsNone(run_spike.parse_cells(CFG, None))
        with self.assertRaisesRegex(ValueError, "unknown cell"):
            run_spike.parse_cells(CFG, ["seer-public-loud"])


if __name__ == "__main__":
    unittest.main()
