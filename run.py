"""Project entry point.  Run:  python run.py <command>

Commands
--------
  phase0    end-to-end Phase 0: npv self-test, EDA report, dummy submission, validate
  npv       run the NPV self-test
  eda       (re)generate reports/eda_report.md
  dummy     write the validator-passing dummy submission into out/
  validate  run the official validator on out/
"""
from __future__ import annotations

import sys

from src import eda, npv, submission, validate
from src import phase1 as _phase1
from src import phase2 as _phase2
from src import phase3 as _phase3
from src import phase5 as _phase5


def cmd_npv() -> bool:
    npv._selftest()
    return True


def cmd_eda() -> bool:
    eda.write_report()
    return True


def cmd_dummy() -> bool:
    submission.build_dummy()
    return True


def cmd_validate() -> bool:
    return validate.run()


def cmd_phase1() -> bool:
    return _phase1.run()


def cmd_phase2() -> bool:
    return _phase2.run()


def cmd_phase3() -> bool:
    return _phase3.run()


def cmd_phase5() -> bool:
    return _phase5.run()


def cmd_phase0() -> bool:
    print("=" * 70, "\nPHASE 0 — setup, EDA, baseline submission, validation\n" + "=" * 70)
    print("\n[1/4] NPV self-test"); npv._selftest()
    print("\n[2/4] EDA report"); eda.write_report()
    print("\n[3/4] Dummy submission"); submission.build_dummy()
    print("\n[4/4] Validate"); ok = validate.run()
    print("\n" + "=" * 70)
    print("PHASE 0 RESULT:", "OK — submittable baseline in out/" if ok else "FAILED — see errors above")
    print("=" * 70)
    return ok


COMMANDS = {
    "phase0": cmd_phase0, "phase1": cmd_phase1, "phase2": cmd_phase2, "phase3": cmd_phase3,
    "phase5": cmd_phase5,
    "npv": cmd_npv, "eda": cmd_eda, "dummy": cmd_dummy, "validate": cmd_validate,
}


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "phase0"
    if cmd not in COMMANDS:
        print(__doc__); return 2
    ok = COMMANDS[cmd]()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
