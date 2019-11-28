MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

.DEFAULT_GOAL := build

SHELL := /bin/bash

# Variables

ECHO ?= echo -e

BUILD_DIR ?= _build
OUTPUT_DIR ?= _output
INSTALL_DIR ?= _install

CMAKE ?= cmake
CMAKE_OPTIONS ?= \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=$(MKFILE_DIR)/$(INSTALL_DIR)
# make CMAKE_OPTIONS=\"-DCMAKE_BUILD_TYPE=Release\ -DCMAKE_INSTALL_PREFIX=_install\"

INIT_OPTIONS ?=
MAKE_OPTIONS ?=

# Functions

define echo
	text="$1"; options="$2"; \
	[ -z "$$options" ] && options="1;33"; \
	$(ECHO) "\033[$${options}m$${text}\033[0m"
endef

define md
	([ -e "$1" ] || (mkdir -p "$1" && $(call echo,MD: $1,33)))
endef

define rm
	[ ! -h "$1" ] && [ ! -e "$1" ] || (rm -r "$1" && $(call echo,RM: $1,33))
endef

# Goals

.PHONY: build
build:
	@$(call md,$(BUILD_DIR))
	@cd $(BUILD_DIR); \
	$(CMAKE) $(CMAKE_OPTIONS) ..; \
	make $(MAKE_OPTIONS)

.PHONY: install
install: build
	@cd $(BUILD_DIR); make install

.PHONY: clean
clean:
	@$(call rm,$(BUILD_DIR))
	@$(call rm,$(OUTPUT_DIR))
	@$(call rm,$(INSTALL_DIR))

.PHONY: print
print:
	@$(call echo,Make $@)
	@echo MKFILE_PATH: $(MKFILE_PATH)
	@echo MKFILE_DIR: $(MKFILE_DIR)
	@echo SHELL: $(SHELL)
	@echo ECHO: $(ECHO)
	@echo CMAKE: $(CMAKE)
	@echo CMAKE_OPTIONS: $(CMAKE_OPTIONS)
	@echo INIT_OPTIONS: $(INIT_OPTIONS)
	@echo MAKE_OPTIONS: $(MAKE_OPTIONS)
