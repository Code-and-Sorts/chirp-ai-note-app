ARCH ?= $(shell uname -m)
SWIFT ?= swiftc
CC ?= clang
BUNDLE := ../CaptureAudio.app
MACOS_DIR := $(BUNDLE)/Contents/MacOS
BINARY := $(MACOS_DIR)/capture_audio
LAUNCHER := $(MACOS_DIR)/disclaim_launcher
PLIST_DEST := $(BUNDLE)/Contents/Info.plist
SOURCE := capture_audio.swift
LAUNCHER_SOURCE := disclaim_launcher.c
PLIST_SOURCE := Info.plist
PYPROJECT := ../../pyproject.toml
VERSION := $(shell awk -F'"' '/^version[[:space:]]*=/ {print $$2; exit}' $(PYPROJECT))

.PHONY: build clean

build: $(BINARY) $(LAUNCHER) $(PLIST_DEST)

$(MACOS_DIR):
	mkdir -p $(MACOS_DIR)

$(BINARY): $(SOURCE) | $(MACOS_DIR)
	$(SWIFT) $(SOURCE) \
		-framework ScreenCaptureKit \
		-framework CoreMedia \
		-framework AVFoundation \
		-target $(ARCH)-apple-macos13 \
		-O \
		-o $(BINARY)
	chmod +x $(BINARY)

$(LAUNCHER): $(LAUNCHER_SOURCE) | $(MACOS_DIR)
	$(CC) $(LAUNCHER_SOURCE) \
		-target $(ARCH)-apple-macos13 \
		-O2 \
		-o $(LAUNCHER)
	chmod +x $(LAUNCHER)

# CFBundleShortVersionString / CFBundleVersion track pyproject.toml so the
# release-tag flow (scripts/update_release_version.py) propagates straight
# into the helper bundle without a second source of truth.
$(PLIST_DEST): $(PLIST_SOURCE) $(PYPROJECT) | $(MACOS_DIR)
	@if [ -z "$(VERSION)" ]; then \
		echo "error: could not read version from $(PYPROJECT)"; exit 1; \
	fi
	sed 's/__CHIRP_VERSION__/$(VERSION)/g' $(PLIST_SOURCE) > $(PLIST_DEST)

clean:
	rm -rf $(BUNDLE)
