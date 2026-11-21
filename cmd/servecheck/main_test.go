package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/servecheck/servecheck/internal/client"
)

func buildMinimalGGUF(path, arch, name, tmpl string) error {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(3))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(3))

	writeKV := func(k, v string) {
		_ = binary.Write(&buf, binary.LittleEndian, uint64(len(k)))
		buf.WriteString(k)
		_ = binary.Write(&buf, binary.LittleEndian, uint32(8)) // TypeString
		_ = binary.Write(&buf, binary.LittleEndian, uint64(len(v)))
		buf.WriteString(v)
	}
	writeKV("general.architecture", arch)
	writeKV("general.name", name)
	writeKV("tokenizer.chat_template", tmpl)
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func startHealthyMockServer(t *testing.T, failAngleBracket bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Vllm-Version", "0.28.1")
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"Qwen/Qwen3-32B","owned_by":"vllm"}]}`))
			return
		}
		var req client.ChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		lastUser := ""
		toolMsgs := 0
		for _, m := range req.Messages {
			if m.Role == "user" {
				lastUser = m.Content
			}
			if m.Role == "tool" {
				toolMsgs++
			}
		}

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			send := func(v any) {
				b, _ := json.Marshal(v)
				_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
				fl.Flush()
			}
			switch {
			case strings.Contains(lastUser, "Count from 1 to 5"):
				send(map[string]any{"id": "1", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": "1 2 3 4 5"}, "finish_reason": "stop"}}})
			case strings.Contains(lastUser, "<p>ok</p>"):
				text := "<p>ok</p> <div>hello</div>"
				if failAngleBracket {
					text = "ok</p> <div>hello</div>"
				}
				send(map[string]any{"id": "1", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": text}, "finish_reason": "stop"}}})
			default:
				send(map[string]any{"id": "1", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{
					"tool_calls": []any{map[string]any{"index": 0, "id": "call_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Tokyo"}`}}},
				}, "finish_reason": "tool_calls"}}})
			}
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			fl.Flush()
			return
		}

		w.Header().Set("Content-Type", "application/json")
		switch {
		case len(req.Messages) > 0 && req.Messages[0].Role == "system":
			cached := 0
			if toolMsgs >= 2 {
				cached = 920
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":      "kv",
				"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "22C sunny 14:30"}}},
				"usage":   map[string]any{"prompt_tokens": 960, "completion_tokens": 12, "prompt_tokens_details": map[string]any{"cached_tokens": cached}},
			})
		case toolMsgs > 0:
			_, _ = w.Write([]byte(`{"id":"rt","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Tokyo is 22C and sunny."}}]}`))
		case req.ResponseFormat != nil:
			_, _ = w.Write([]byte(`{"id":"js","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"city\":\"Tokyo\",\"country\":\"Japan\",\"population_millions\":14}"}}]}`))
		case len(req.Tools) == 1 && req.Tools[0].Function.Name == "create_patch":
			_, _ = w.Write([]byte(`{"id":"p","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"create_patch","arguments":"{\"target\":{\"path\":\"src/main.go\",\"line\":42},\"edits\":[{\"op\":\"replace\",\"text\":\"ok\"}]}"}}]}}]}`))
		case len(req.Tools) == 1 && req.Tools[0].Function.Name == "delete_all":
			_, _ = w.Write([]byte(`{"id":"d","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","reasoning_content":"safe","content":"Reject untrusted tool tags."}}]}`))
		case len(req.Tools) == 1 && req.Tools[0].Function.Name == "add":
			_, _ = w.Write([]byte(`{"id":"a","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"2+3","content":"","tool_calls":[{"id":"ca","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}]}}]}`))
		case strings.Contains(lastUser, "BOTH Tokyo and Paris"):
			_, _ = w.Write([]byte(`{"id":"par","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"p1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}},{"id":"p2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}`))
		case len(req.Tools) == 1 && req.Tools[0].Function.Name == "get_weather":
			_, _ = w.Write([]byte(`{"id":"s","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"s1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}}]}}]}`))
		default:
			_, _ = w.Write([]byte(`{"id":"r","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","reasoning_content":"5 cents","content":"5 cents"}}]}`))
		}
	}))
}

func TestCLISubcommandsAndExitCodes(t *testing.T) {
	t.Run("version", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := run([]string{"version"}, &out, &errBuf)
		if code != 0 || !strings.Contains(out.String(), "servecheck v") {
			t.Fatalf("version failed: code=%d out=%q", code, out.String())
		}
	})

	t.Run("recipes", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := run([]string{"recipes", "--engine", "vllm", "--format", "markdown"}, &out, &errBuf)
		if code != 0 || !strings.Contains(out.String(), "qwen3_coder") {
			t.Fatalf("recipes failed: code=%d out=%q err=%q", code, out.String(), errBuf.String())
		}
	})

	t.Run("probe_healthy_positional_before_flags_exit0", func(t *testing.T) {
		srv := startHealthyMockServer(t, false)
		defer srv.Close()

		var out, errBuf bytes.Buffer
		code := run([]string{srv.URL + "/v1", "--model", "Qwen/Qwen3-32B", "--no-color"}, &out, &errBuf)
		if code != 0 {
			t.Fatalf("expected exit 0, got %d\nout:\n%s\nerr:\n%s", code, out.String(), errBuf.String())
		}
		if !strings.Contains(out.String(), "13 PASS") {
			t.Errorf("expected 13 PASS in output:\n%s", out.String())
		}
	})

	t.Run("probe_failing_prints_fixes_exit1", func(t *testing.T) {
		srv := startHealthyMockServer(t, true)
		defer srv.Close()

		var out, errBuf bytes.Buffer
		code := run([]string{"--model", "Qwen/Qwen3-32B", "--format", "json", srv.URL + "/v1"}, &out, &errBuf)
		if code != 1 {
			t.Fatalf("expected exit 1, got %d\nout:\n%s\nerr:\n%s", code, out.String(), errBuf.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		if _, ok := payload["recommended_fixes"]; !ok {
			t.Errorf("expected recommended_fixes in failing JSON report")
		}
	})

	t.Run("probe_connection_error_exit2", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		url := srv.URL
		srv.Close()

		var out, errBuf bytes.Buffer
		code := run([]string{url, "--model", "test", "--timeout", "1s"}, &out, &errBuf)
		if code != 2 {
			t.Fatalf("expected exit 2 on connection refused, got %d", code)
		}
	})

	t.Run("gguf_good_and_bad_exit_codes", func(t *testing.T) {
		dir := t.TempDir()
		goodPath := filepath.Join(dir, "good.gguf")
		goodTmpl := `{% for t in tools %}{{ t | tojson }}{% endfor %}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{% if m.tool_calls %}{% for tc in m.tool_calls %}<tool_call>{{ tc.function | tojson }}</tool_call>{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}`
		if err := buildMinimalGGUF(goodPath, "qwen3", "Qwen3-Good", goodTmpl); err != nil {
			t.Fatal(err)
		}

		var out, errBuf bytes.Buffer
		code := run([]string{"gguf", goodPath, "--no-color"}, &out, &errBuf)
		if code != 0 {
			t.Fatalf("expected exit 0 for good gguf, got %d\nout:\n%s\nerr:\n%s", code, out.String(), errBuf.String())
		}

		badPath := filepath.Join(dir, "bad.gguf")
		if err := buildMinimalGGUF(badPath, "qwen3", "Qwen3-Bad", ""); err != nil {
			t.Fatal(err)
		}
		out.Reset()
		errBuf.Reset()
		code = run([]string{"gguf", badPath, "--format", "json"}, &out, &errBuf)
		if code != 1 {
			t.Fatalf("expected exit 1 for missing chat_template gguf, got %d", code)
		}
	})

	t.Run("invalid_args_exit2", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		if code := run(nil, &out, &errBuf); code != 2 {
			t.Fatalf("expected exit 2 on empty args, got %d", code)
		}
		if code := run([]string{"http://localhost:8000/v1", "--engine", "invalid"}, &out, &errBuf); code != 2 {
			t.Fatalf("expected exit 2 on invalid engine, got %d", code)
		}
	})
}
