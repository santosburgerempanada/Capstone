"""Corre las instancias I-VI y los análisis de sensibilidad definidos en config/experiments.yaml.
Guarda una tabla por experimento en results/ (análogas a las Tablas 7, 8 y 9 y la Fig. 8)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.yaml")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
