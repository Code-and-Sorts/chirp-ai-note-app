ARCH ?= $(shell uname -m)
SWIFT ?= swiftc
SIGN_IDENTITY := $(or $(CHIRP_CODESIGN_IDENTITY),-)
BUNDLE := ../Chirp.app
MACOS_DIR := $(BUNDLE)/Contents/MacOS
RESOURCES_DIR := $(BUNDLE)/Contents/Resources
BINARY := $(MACOS_DIR)/capture_audio
PLIST_DEST := $(BUNDLE)/Contents/Info.plist
ICON_DEST := $(RESOURCES_DIR)/AppIcon.icns
SOURCE := capture_audio.swift
PLIST_SOURCE := Info.plist
ICON_SOURCE := AppIcon.png
PYPROJECT := ../../pyproject.toml
VERSION := $(shell awk -F'"' '/^version[[:space:]]*=/ {print $$2; exit}' $(PYPROJECT))

.PHONY: build clean codesign

build: $(BINARY) $(PLIST_DEST) $(ICON_DEST) codesign

codesign:
	codesign --force --deep --sign "$(SIGN_IDENTITY)" $(BUNDLE)

$(MACOS_DIR):
	mkdir -p $(MACOS_DIR)

$(RESOURCES_DIR):
	mkdir -p $(RESOURCES_DIR)

$(ICON_DEST): $(ICON_SOURCE) | $(RESOURCES_DIR)
	iconset=$$(mktemp -d)/AppIcon.iconset; mkdir -p $$iconset; \
	for size in 16 32 128 256 512; do \
		sips -z $$size $$size $(ICON_SOURCE) --out $$iconset/icon_$${size}x$${size}.png >/dev/null; \
		double=$$((size * 2)); \
		sips -z $$double $$double $(ICON_SOURCE) --out $$iconset/icon_$${size}x$${size}@2x.png >/dev/null; \
	done; \
	iconutil -c icns $$iconset -o $(ICON_DEST); \
	rm -rf $$iconset

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
