def main() -> int:
    from importlib.metadata import version

    print(f"chirpd {version('chirp-notes-ai')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
