ARCH ?= $(shell uname -m)
SWIFT ?= swiftc
BUNDLE := ../CaptureAudio.app
MACOS_DIR := $(BUNDLE)/Contents/MacOS
BINARY := $(MACOS_DIR)/capture_audio
PLIST_DEST := $(BUNDLE)/Contents/Info.plist
SOURCE := capture_audio.swift
PLIST_SOURCE := Info.plist

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

$(PLIST_DEST): $(PLIST_SOURCE) | $(MACOS_DIR)
	cp $(PLIST_SOURCE) $(PLIST_DEST)

clean:
	rm -rf $(BUNDLE)
