BINARY   := bin/servecheck
NAME     := servecheck
VERSION  ?= 0.1.0
LDFLAGS  := -s -w -X main.version=$(VERSION)
PLATFORMS := darwin/arm64 darwin/amd64 linux/amd64 linux/arm64

.PHONY: all build test vet clean install cross package

all: vet test build

build:
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/servecheck

test:
	go test -race -count=1 -v ./...

vet:
	go vet ./...

install:
	CGO_ENABLED=0 go install -trimpath -ldflags "$(LDFLAGS)" ./cmd/servecheck

cross:
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
		os=$${p%/*}; arch=$${p#*/}; \
		echo "Building $(NAME)-$$os-$$arch..."; \
		CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -trimpath -ldflags "$(LDFLAGS)" -o "dist/$(NAME)-$$os-$$arch" ./cmd/servecheck; \
	done

package:
	@rm -rf dist
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
		os=$${p%/*}; arch=$${p#*/}; \
		stage="dist/.stage-$$os-$$arch"; \
		archive="$(NAME)_v$(VERSION)_$${os}_$${arch}.tar.gz"; \
		mkdir -p "$$stage"; \
		CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build -trimpath -ldflags "$(LDFLAGS)" -o "$$stage/$(NAME)" ./cmd/servecheck; \
		cp README.md LICENSE "$$stage/"; \
		tar -C "$$stage" -czf "dist/$$archive" "$(NAME)" README.md LICENSE; \
		rm -rf "$$stage"; \
	done
	@cd dist && (command -v sha256sum >/dev/null 2>&1 && sha256sum *.tar.gz || shasum -a 256 *.tar.gz) > checksums.txt
	@cat dist/checksums.txt

clean:
	rm -rf bin dist coverage.out
