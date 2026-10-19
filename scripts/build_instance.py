import argparse
from pathlib import Path

from ebus.config import load_config
from ebus.data.build_instance import build_instance, save_instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    inst = build_instance(cfg)
    out = args.out or Path("data/processed") / f"{cfg['case']['name']}.pkl"
    save_instance(inst, out)
    print(f"{len(inst.trips)} viajes, {len(inst.depots)} depósitos -> {out}")


if __name__ == "__main__":
    main()
