BINARY   := bin/servecheck
VERSION  ?= 0.1.0
LDFLAGS  := -s -w -X main.version=$(VERSION)

.PHONY: all build test vet clean install cross

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
	mkdir -p dist
	CGO_ENABLED=0 GOOS=darwin  GOARCH=arm64 go build -trimpath -ldflags "$(LDFLAGS)" -o dist/servecheck-darwin-arm64  ./cmd/servecheck
	CGO_ENABLED=0 GOOS=linux   GOARCH=amd64 go build -trimpath -ldflags "$(LDFLAGS)" -o dist/servecheck-linux-amd64   ./cmd/servecheck
	CGO_ENABLED=0 GOOS=linux   GOARCH=arm64 go build -trimpath -ldflags "$(LDFLAGS)" -o dist/servecheck-linux-arm64   ./cmd/servecheck

clean:
	rm -rf bin dist coverage.out
