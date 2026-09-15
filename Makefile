# =============================================================================
# factreasoner — Docker build & publish
# =============================================================================
#
# Usage:
#   make build                        # build default Python version (no local load)
#   make build LOAD=true              # build + load image into local Docker daemon
#   make build PYTHON_VERSION=312     # build a specific Python version
#   make push                         # build + push default version
#   make build-all                    # build all PYTHON_VERSIONS
#   make push-all                     # build + push all PYTHON_VERSIONS
#
# Optional:
#   LOAD     — load image into local Docker daemon after build (default: false).
#              Example: make build LOAD=true
#   BUILDER  — name of a docker buildx builder to use (default: current context).
#              Example: make build BUILDER=zrl2
#              Example: make build BUILDER=zrl2 LOAD=true
#              Use BUILDER_OUTPUT to override the output mode entirely, e.g.:
#                make build BUILDER=zrl2 BUILDER_OUTPUT=--push
#
# Required env vars for push targets:
#   REGISTRY_PASSWORD  — ICR API key
#   REGISTRY_USERNAME  — ICR username (iamapikey for ICR)
# =============================================================================

REGISTRY        ?= icr.io
NAMESPACE       ?= drl-nextgen
IMAGE_NAME      ?= factreasoner
FULL_IMAGE      := $(REGISTRY)/$(NAMESPACE)/$(IMAGE_NAME)

PYTHON_VERSIONS ?= 311 312
PYTHON_VERSION  ?= 311

GIT_SHA         := $(shell git rev-parse --short HEAD)
GIT_TAG         := $(shell git tag --points-at HEAD | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$' | head -1)
VARIANT         := py$(PYTHON_VERSION)

ifneq ($(GIT_TAG),)
SEMVER_TAG      := $(GIT_TAG)-$(VARIANT)
endif

SHA_TAG         := $(GIT_SHA)-$(VARIANT)
LATEST_TAG      := latest-$(VARIANT)

TAG_FLAGS       := --tag $(FULL_IMAGE):$(SHA_TAG) \
                   --tag $(FULL_IMAGE):$(LATEST_TAG)
ifneq ($(GIT_TAG),)
TAG_FLAGS       += --tag $(FULL_IMAGE):$(SEMVER_TAG)
endif

PLATFORM        ?= linux/amd64
BUILDER         ?=

# LOAD=true adds --load to pull the image into the local Docker daemon.
# Default is false — the result stays in the BuildKit cache (no tarball transfer).
LOAD            ?= false
_LOAD_FLAG      := $(if $(filter true,$(LOAD)),--load,)

ifdef BUILDER
  _BUILDER_FLAG  := --builder $(BUILDER)
  BUILDER_OUTPUT ?= $(_LOAD_FLAG)
else
  _BUILDER_FLAG  :=
  BUILDER_OUTPUT ?= $(_LOAD_FLAG)
endif

# =============================================================================

.PHONY: build push build-all push-all login clean help

## Build the image for a single PYTHON_VERSION
build:
	docker buildx build \
		--platform $(PLATFORM) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		$(TAG_FLAGS) \
		$(_BUILDER_FLAG) \
		$(BUILDER_OUTPUT) \
		-f Dockerfile \
		.

## Push the image for a single PYTHON_VERSION (builds + pushes directly to registry)
push:
	$(MAKE) build BUILDER_OUTPUT=--push

## Build all variants in PYTHON_VERSIONS
build-all:
	@for pyver in $(PYTHON_VERSIONS); do \
		echo "==> Building py$$pyver"; \
		$(MAKE) build PYTHON_VERSION=$$pyver; \
	done

## Build and push all variants in PYTHON_VERSIONS
push-all:
	@for pyver in $(PYTHON_VERSIONS); do \
		echo "==> Pushing py$$pyver"; \
		$(MAKE) push PYTHON_VERSION=$$pyver; \
	done

## Log in to the container registry
login:
	@echo "$(REGISTRY_PASSWORD)" | \
		docker login $(REGISTRY) -u "$(REGISTRY_USERNAME)" --password-stdin

## Remove locally built images for the current variant
clean:
	-docker rmi $(FULL_IMAGE):$(SHA_TAG) 2>/dev/null
	-docker rmi $(FULL_IMAGE):$(LATEST_TAG) 2>/dev/null
ifneq ($(GIT_TAG),)
	-docker rmi $(FULL_IMAGE):$(SEMVER_TAG) 2>/dev/null
endif

## Print resolved variables (dry run)
help:
	@echo ""
	@echo "  FULL_IMAGE     : $(FULL_IMAGE)"
	@echo "  PLATFORM       : $(PLATFORM)"
	@echo "  BUILDER        : $(if $(BUILDER),$(BUILDER),(default context))"
	@echo "  BUILDER_OUTPUT : $(BUILDER_OUTPUT)"
	@echo "  VARIANT        : $(VARIANT)"
	@echo "  SHA_TAG        : $(SHA_TAG)"
	@echo "  LATEST_TAG     : $(LATEST_TAG)"
	@echo "  SEMVER_TAG     : $(if $(GIT_TAG),$(SEMVER_TAG),(not on a version tag))"
	@echo ""
	@echo "Targets: build | push | build-all | push-all | login | clean | help"
	@echo ""
