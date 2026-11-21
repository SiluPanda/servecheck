package client

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNormalizeBaseURL(t *testing.T) {
	tests := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"http://localhost:8000", "http://localhost:8000/v1", false},
		{"http://localhost:8000/", "http://localhost:8000/v1", false},
		{"http://localhost:8000/v1", "http://localhost:8000/v1", false},
		{"http://localhost:8000/v1/", "http://localhost:8000/v1", false},
		{"https://api.example.com/custom/openai/", "https://api.example.com/custom/openai", false},
		{"127.0.0.1:11434", "http://127.0.0.1:11434/v1", false},
		{"", "", true},
		{"http://", "", true},
	}

	for _, tc := range tests {
		got, err := NormalizeBaseURL(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("NormalizeBaseURL(%q): expected error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("NormalizeBaseURL(%q): unexpected error: %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("NormalizeBaseURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestDetectEngine(t *testing.T) {
	t.Run("vllm header", func(t *testing.T) {
		h := http.Header{"X-Vllm-Version": []string{"0.28.1"}, "Server": []string{"uvicorn"}}
		got := DetectEngine(h, nil, "auto")
		if got.Engine != "vllm" || got.Version != "0.28.1" {
			t.Fatalf("got %+v, want vllm 0.28.1", got)
		}
	})

	t.Run("llama-server header", func(t *testing.T) {
		h := http.Header{"Server": []string{"llama.cpp/b5120"}}
		got := DetectEngine(h, nil, "")
		if got.Engine != "llama-server" || got.Version != "b5120" {
			t.Fatalf("got %+v, want llama-server b5120", got)
		}
	})

	t.Run("sglang header", func(t *testing.T) {
		h := http.Header{"X-Sglang-Version": []string{"0.4.7"}}
		got := DetectEngine(h, nil, "auto")
		if got.Engine != "sglang" || got.Version != "0.4.7" {
			t.Fatalf("got %+v, want sglang 0.4.7", got)
		}
	})

	t.Run("ollama models owned_by", func(t *testing.T) {
		models := &ModelsResponse{Data: []ModelItem{{ID: "qwen3:32b", OwnedBy: "library"}}}
		got := DetectEngine(http.Header{}, models, "auto")
		if got.Engine != "ollama" {
			t.Fatalf("got %+v, want ollama", got)
		}
	})

	t.Run("mlx models owned_by", func(t *testing.T) {
		models := &ModelsResponse{Data: []ModelItem{{ID: "mlx-community/Qwen3-8B", OwnedBy: "mlx-lm"}}}
		got := DetectEngine(http.Header{}, models, "auto")
		if got.Engine != "mlx" {
			t.Fatalf("got %+v, want mlx", got)
		}
	})

	t.Run("override takes precedence", func(t *testing.T) {
		h := http.Header{"X-Vllm-Version": []string{"0.28.1"}}
		got := DetectEngine(h, nil, "llama-server")
		if got.Engine != "llama-server" || got.DetectedFrom != "flag:--engine" {
			t.Fatalf("got %+v, want override llama-server", got)
		}
	})
}

func TestMessageReasoningPresence(t *testing.T) {
	var m1 Message
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":"hi","reasoning":"thought"}`), &m1); err != nil {
		t.Fatal(err)
	}
	if !m1.HasReasoning || m1.HasReasoningContent || m1.Reasoning != "thought" {
		t.Fatalf("unexpected m1: %+v", m1)
	}

	var m2 Message
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":"hi","reasoning_content":"step 1"}`), &m2); err != nil {
		t.Fatal(err)
	}
	if m2.HasReasoning || !m2.HasReasoningContent || m2.ReasoningContent != "step 1" {
		t.Fatalf("unexpected m2: %+v", m2)
	}
}

func TestStreamAndAuthErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"unauthorized"}`))
			return
		}
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			_, _ = w.Write([]byte(": keepalive\n\n"))
			_, _ = w.Write([]byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"hel"}}]}` + "\n\n"))
			fl.Flush()
			_, _ = w.Write([]byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			fl.Flush()
		}
	}))
	defer srv.Close()

	// Auth failure check.
	unauth, err := New(srv.URL, "wrong", 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	_, err = unauth.Models(context.Background())
	var authErr *AuthError
	if !errors.As(err, &authErr) {
		t.Fatalf("expected AuthError, got %v", err)
	}

	// Stream parsing check.
	okClient, err := New(srv.URL, "secret", 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	sr, err := okClient.ChatCompletionStream(context.Background(), ChatRequest{
		Model:    "test",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !sr.SawDone || len(sr.Chunks) != 2 || len(sr.ParseErrors) != 0 {
		t.Fatalf("unexpected stream result: done=%v chunks=%d errs=%v", sr.SawDone, len(sr.Chunks), sr.ParseErrors)
	}
}
