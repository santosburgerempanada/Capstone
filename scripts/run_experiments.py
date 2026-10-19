import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.yaml")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
