"""Runs the demo PD project and fails unless every block ends green. Run
from a folder set up by `modelmaker-demo`, outside the checkout, so
`import modelmaker` is the installed wheel. A file rather than a stdin
heredoc: the runner's worker processes re-import __main__ by path."""

from pathlib import Path

import modelmaker.api  # registers every block, as the server does
from modelmaker.session import ProjectSession

if __name__ == "__main__":
    session = ProjectSession(recovery_path=None)
    session.load(Path("projects/demo_pd_model/model.json"))
    session.runner.run_all()
    not_green = [b.name for bid, b in session.graph.blocks.items() if session.runner.status(bid) != "green"]
    assert not not_green, f"demo blocks not green: {not_green}"
    print("demo pipeline ran green from the installed wheel")
