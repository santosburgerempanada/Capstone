"""Particiona los datos crudos en subproblemas y los guarda en data/processed/<particion>/."""


import argparse

from ebus.config import load_config
from ebus.particion.subproblemas import generar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/particion.yaml")
    parser.add_argument("--out", default=None, help="default: data/processed/<particion.nombre>")
    parser.add_argument("--piloto", default=None, help="terminal del Excel piloto (default: el menos saturado)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fname = generar(cfg, out_dir=args.out, piloto=args.piloto)
    print(f"piloto: {fname}")


if __name__ == "__main__":
    main()
