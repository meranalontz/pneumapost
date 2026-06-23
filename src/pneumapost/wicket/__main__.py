from __future__ import annotations

import argh

from .appraise import appraise
from .export import export
from .predict import predict
from .train import train


def main() -> None:
    parser = argh.ArghParser(prog="pneumapost")
    parser.add_commands([train, appraise, predict, export])
    parser.dispatch()


if __name__ == "__main__":
    main()
