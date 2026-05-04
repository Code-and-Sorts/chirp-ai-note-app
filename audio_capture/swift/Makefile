ARCH ?= $(shell uname -m)
SWIFT ?= swiftc
BUNDLE := ../CaptureAudio.app
MACOS_DIR := $(BUNDLE)/Contents/MacOS
BINARY := $(MACOS_DIR)/capture_audio
PLIST_DEST := $(BUNDLE)/Contents/Info.plist
SOURCE := capture_audio.swift
PLIST_SOURCE := Info.plist
PYPROJECT := ../../pyproject.toml
VERSION := $(shell awk -F'"' '/^version[[:space:]]*=/ {print $$2; exit}' $(PYPROJECT))

.PHONY: build clean

build: $(BINARY) $(PLIST_DEST)

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
