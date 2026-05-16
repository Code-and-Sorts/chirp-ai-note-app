from importlib.metadata import version


def main() -> None:
    print(f"chirpd {version('chirp-notes-ai')}")


if __name__ == "__main__":
    main()
