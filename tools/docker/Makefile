SOURCES=$(wildcard Dockerfile.*)
PHONY_TARGETS=$(subst Dockerfile.,,$(SOURCES))
ROOT_DIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TOOLS_DIR=$(ROOT_DIR)/..

.PHONY:
all: $(PHONY_TARGETS)

.PHONY:
$(PHONY_TARGETS): %: .Dockerfile.%.stamp

.Dockerfile.%.stamp: Dockerfile.%
	docker build -t $* -f $(ROOT_DIR)/$^ $(TOOLS_DIR)
	touch $@

.PHONY:
clean:
	rm .Dockerfile.*.stamp -f
