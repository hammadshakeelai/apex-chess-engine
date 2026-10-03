"""
Unit tests for the High-Performance C++ Core Engine (bin/apex_engine.exe)
"""

import os
import subprocess
import pytest

CPP_EXE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "apex_engine.exe")
MINGW_BIN = "C:\\msys64\\mingw64\\bin"


@pytest.mark.skipif(not os.path.exists(CPP_EXE), reason="C++ binary not compiled")
class TestCPPEngine:

    def _run_cpp(self, commands: list[str]) -> str:
        env = os.environ.copy()
        if MINGW_BIN not in env.get("PATH", ""):
            env["PATH"] = MINGW_BIN + ";" + env.get("PATH", "")

        input_str = "\n".join(commands) + "\nquit\n"
        res = subprocess.run(
            [CPP_EXE],
            input=input_str,
            text=True,
            capture_output=True,
            timeout=15,
        )
        return res.stdout

    def test_version_banner(self):
        env = os.environ.copy()
        env["PATH"] = MINGW_BIN + ";" + env.get("PATH", "")
        res = subprocess.run([CPP_EXE, "--version"], text=True, capture_output=True, env=env)
        assert res.returncode == 0
        assert "ApexChess 1.0" in res.stdout

    def test_perft_3(self):
        env = os.environ.copy()
        env["PATH"] = MINGW_BIN + ";" + env.get("PATH", "")
        res = subprocess.run([CPP_EXE, "--perft", "3"], text=True, capture_output=True, env=env)
        assert res.returncode == 0
        assert "Total Nodes Evaluated: 8902" in res.stdout

    def test_perft_4(self):
        env = os.environ.copy()
        env["PATH"] = MINGW_BIN + ";" + env.get("PATH", "")
        res = subprocess.run([CPP_EXE, "--perft", "4"], text=True, capture_output=True, env=env)
        assert res.returncode == 0
        assert "Total Nodes Evaluated: 197281" in res.stdout

    def test_uci_protocol_flow(self):
        out = self._run_cpp(["uci", "isready"])
        assert "ApexChess 1.0 (C++ Core" in out
        assert "uciok" in out
        assert "readyok" in out

    def test_mate_in_one_tactical_solver(self):
        # Scholar's mate position: Qxf7#
        out = self._run_cpp([
            "position fen r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 5",
            "go depth 4",
        ])
        assert "bestmove f3f7" in out
