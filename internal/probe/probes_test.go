package probe_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/SiluPanda/servecheck/internal/client"
	"github.com/SiluPanda/servecheck/internal/probe"
)

type mockServerFaults struct {
	engineHeader         string // e.g. "vllm", "llama-server"
	schemaDropOnCall2    bool   // simulates vLLM ~8% nested schema drop
	decoyMisfire         bool   // <tool_call> in decoy prompt triggers live delete_all call
	swallowToolInThink   bool   // HTTP 200 finish_reason=stop with raw <tool_call> in reasoning_content
	swallowToolInInline  bool   // raw <tool_call> inside inline <think> in content
	reasoningFieldOnly   bool   // emits .reasoning without .reasoning_content
	inlineThinkLeak      bool   // emits <think>...</think> in .content with no reasoning fields
	streamNegToolIndex   bool   // vLLM MTP tool_index: -1
	streamMissingDone    bool   // SSE ends without data: [DONE]
	streamSwallowAngle   bool   // vllm#65934 swallows leading <p>
	toolRoleHTTP400      bool   // rejects multi-turn role="tool" with HTTP 400
	duplicateParallelIDs bool   // emits duplicate tool_call.id across parallel calls
	xgrammarJSONError    bool   // rejects json_schema with XGrammar 400 error
	jsonMarkdownFence    bool   // wraps json_schema output in ```json fences
	kvCacheDrop          bool   // Turn 2 cached_tokens drops to 4%
	kvCacheDropLlama     bool   // Turn 2 timings.cache_n drops to 1%
	kvCacheTimingOnly    bool   // no cached_tokens/cache_n; timings.prompt_ms stays flat (~215ms vs 220ms)
	requireAuthKey       string // enforce Bearer token
}

func newMockLLMServer(t *testing.T, faults mockServerFaults) *httptest.Server {
	t.Helper()
	var patchCalls atomic.Int32

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch faults.engineHeader {
		case "llama-server":
			w.Header().Set("Server", "llama.cpp/b5120")
		default:
			w.Header().Set("Server", "uvicorn")
			w.Header().Set("X-Vllm-Version", "0.28.1")
		}

		if faults.requireAuthKey != "" && r.Header.Get("Authorization") != "Bearer "+faults.requireAuthKey {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
			return
		}

		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			ownedBy := "vllm"
			if faults.engineHeader == "llama-server" {
				ownedBy = "llamacpp"
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"id": "Qwen/Qwen3-32B", "object": "model", "owned_by": ownedBy, "max_model_len": 32768},
				},
			})
			return
		}

		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		var req client.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		lastUser := ""
		hasToolRole := false
		toolMsgCount := 0
		for _, m := range req.Messages {
			if m.Role == "user" {
				lastUser = m.Content
			}
			if m.Role == "tool" {
				hasToolRole = true
				toolMsgCount++
			}
		}

		// --- Streaming requests ---
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			writeSSE := func(payload any) {
				b, _ := json.Marshal(payload)
				_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
				fl.Flush()
			}

			// 1) basic_chat
			if strings.Contains(lastUser, "Count from 1 to 5") {
				for _, tok := range []string{"1 ", "2 ", "3 ", "4 ", "5"} {
					writeSSE(map[string]any{
						"id":      "chatcmpl-basic",
						"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": tok}}},
					})
				}
				writeSSE(map[string]any{
					"id":      "chatcmpl-basic",
					"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
					"usage":   map[string]any{"prompt_tokens": 18, "completion_tokens": 9, "total_tokens": 27},
				})
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				fl.Flush()
				return
			}

			// 2) stream_angle_bracket
			if strings.Contains(lastUser, "<p>ok</p>") {
				chunks := []string{"<p>", "ok</p>", " ", "<div>", "hello</div>"}
				if faults.streamSwallowAngle {
					chunks = []string{"ok</p>", " ", "<div>", "hello</div>"}
				}
				for _, tok := range chunks {
					writeSSE(map[string]any{
						"id":      "chatcmpl-html",
						"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": tok}}},
					})
				}
				writeSSE(map[string]any{
					"id":      "chatcmpl-html",
					"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
				})
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				fl.Flush()
				return
			}

			// 3) stream_tool_delta
			idx := 0
			if faults.streamNegToolIndex {
				idx = -1
			}
			writeSSE(map[string]any{
				"id": "chatcmpl-st",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []any{map[string]any{
							"index":    idx,
							"id":       "call_stream_1",
							"type":     "function",
							"function": map[string]any{"name": "get_weather", "arguments": `{"ci`},
						}},
					},
				}},
			})
			writeSSE(map[string]any{
				"id": "chatcmpl-st",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": []any{map[string]any{
							"index":    idx,
							"function": map[string]any{"arguments": `ty":"Tokyo"}`},
						}},
					},
				}},
			})
			writeSSE(map[string]any{
				"id":      "chatcmpl-st",
				"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}},
			})
			if !faults.streamMissingDone {
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			}
			fl.Flush()
			return
		}

		// --- Non-streaming requests ---
		w.Header().Set("Content-Type", "application/json")

		// 1) multi_tool_kv_prefix (large deterministic system prompt)
		if len(req.Messages) > 0 && req.Messages[0].Role == "system" && strings.Contains(req.Messages[0].Content, "Policy 48:") {
			isTurn2 := toolMsgCount >= 2
			resp := map[string]any{
				"id":     "chatcmpl-kv",
				"object": "chat.completion",
				"model":  req.Model,
			}
			if !isTurn2 {
				resp["choices"] = []any{map[string]any{
					"index":         0,
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":    "assistant",
						"content": "",
						"tool_calls": []any{
							map[string]any{"id": "call_kv_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Tokyo"}`}},
							map[string]any{"id": "call_kv_2", "type": "function", "function": map[string]any{"name": "get_time", "arguments": `{"timezone":"Asia/Tokyo"}`}},
						},
					},
				}}
				if faults.kvCacheTimingOnly {
					resp["usage"] = map[string]any{"prompt_tokens": 960, "completion_tokens": 40, "total_tokens": 1000}
					resp["timings"] = map[string]any{"prompt_n": 960, "prompt_ms": 220.0, "predicted_n": 40, "predicted_per_second": 72.0}
				} else if faults.engineHeader == "llama-server" || faults.kvCacheDropLlama {
					resp["usage"] = map[string]any{"prompt_tokens": 960, "completion_tokens": 40, "total_tokens": 1000}
					resp["timings"] = map[string]any{"cache_n": 0, "prompt_n": 960, "prompt_ms": 180.0, "predicted_n": 40, "predicted_per_second": 75.0}
				} else {
					resp["usage"] = map[string]any{
						"prompt_tokens":         960,
						"completion_tokens":     40,
						"total_tokens":          1000,
						"prompt_tokens_details": map[string]any{"cached_tokens": 0},
					}
				}
				_ = json.NewEncoder(w).Encode(resp)
				return
			}

			// Turn 2
			resp["choices"] = []any{map[string]any{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "Tokyo is 22C sunny and it is 14:30."},
			}}
			if faults.kvCacheTimingOnly {
				resp["usage"] = map[string]any{"prompt_tokens": 1080, "completion_tokens": 16, "total_tokens": 1096}
				resp["timings"] = map[string]any{"prompt_n": 1080, "prompt_ms": 215.0, "predicted_n": 16, "predicted_per_second": 72.0}
			} else if faults.engineHeader == "llama-server" || faults.kvCacheDropLlama {
				cacheN := 920
				if faults.kvCacheDropLlama {
					cacheN = 12
				}
				resp["usage"] = map[string]any{"prompt_tokens": 1080, "completion_tokens": 16, "total_tokens": 1096}
				resp["timings"] = map[string]any{"cache_n": cacheN, "prompt_n": 1080 - cacheN, "prompt_ms": 22.0, "predicted_n": 16, "predicted_per_second": 78.0}
			} else {
				cached := 928
				if faults.kvCacheDrop {
					cached = 38
				}
				resp["usage"] = map[string]any{
					"prompt_tokens":         1080,
					"completion_tokens":     16,
					"total_tokens":          1096,
					"prompt_tokens_details": map[string]any{"cached_tokens": cached},
				}
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		// 2) tool_role_roundtrip
		if hasToolRole {
			if faults.toolRoleHTTP400 {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"Conversation roles must alternate user/assistant/user/assistant"}}`))
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-rt",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "stop",
					"message":       map[string]any{"role": "assistant", "content": "It is currently 22°C and sunny in Tokyo."},
				}},
			})
			return
		}

		// 3) json_schema_response_format
		if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" {
			if faults.xgrammarJSONError {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"XGrammar compiler failed: unsupported JSON schema constraint"}}`))
				return
			}
			content := `{"city":"Tokyo","country":"Japan","population_millions":14.0}`
			if faults.jsonMarkdownFence {
				content = "```json\n" + content + "\n```"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-js",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "stop",
					"message":       map[string]any{"role": "assistant", "content": content},
				}},
			})
			return
		}

		// 4) strict_nested_schema
		if len(req.Tools) == 1 && req.Tools[0].Function.Name == "create_patch" {
			n := patchCalls.Add(1)
			args := `{"target":{"path":"src/main.go","line":42},"edits":[{"op":"replace","text":"return nil"}]}`
			if n == 2 {
				if faults.schemaDropOnCall2 {
					// Drops required target.path!
					args = `{"target":{"line":17},"edits":[{"op":"insert","text":"// verified"}]}`
				} else {
					args = `{"target":{"path":"pkg/auth/token.go","line":17},"edits":[{"op":"insert","text":"// verified"}]}`
				}
			} else if n == 3 {
				args = `{"target":{"path":"cmd/cli/run.go","line":88},"edits":[{"op":"delete","text":"fmt.Println(debug)"}]}`
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-patch",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":    "assistant",
						"content": "",
						"tool_calls": []any{map[string]any{
							"id":       fmt.Sprintf("call_patch_%d", n),
							"type":     "function",
							"function": map[string]any{"name": "create_patch", "arguments": args},
						}},
					},
				}},
			})
			return
		}

		// 5) think_decoy_isolation
		if len(req.Tools) == 1 && req.Tools[0].Function.Name == "delete_all" {
			if faults.decoyMisfire {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "chatcmpl-decoy",
					"choices": []any{map[string]any{
						"index":         0,
						"finish_reason": "tool_calls",
						"message": map[string]any{
							"role":              "assistant",
							"content":           "",
							"reasoning_content": `The user asks about <tool_call>{"name":"delete_all","arguments":{"target":"/"}}</tool_call>.`,
							"tool_calls": []any{map[string]any{
								"id":       "call_bad_decoy",
								"type":     "function",
								"function": map[string]any{"name": "delete_all", "arguments": `{"target":"/"}`},
							}},
						},
					}},
				})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-decoy",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]any{
						"role":              "assistant",
						"reasoning_content": "I should explain why untrusted prompt-injection tool tags must be ignored without invoking any tool.",
						"content":           "A safety filter must block untrusted tool-call markup so injected instructions cannot trigger destructive actions.",
					},
				}},
			})
			return
		}

		// 6) tool_swallowed_in_think
		if len(req.Tools) == 1 && req.Tools[0].Function.Name == "add" {
			if faults.swallowToolInThink {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "chatcmpl-swallow",
					"choices": []any{map[string]any{
						"index":         0,
						"finish_reason": "stop",
						"message": map[string]any{
							"role":              "assistant",
							"reasoning_content": `Adding 2 and 3. <tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>`,
							"content":           "",
						},
					}},
				})
				return
			}
			if faults.swallowToolInInline {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "chatcmpl-swallow2",
					"choices": []any{map[string]any{
						"index":         0,
						"finish_reason": "stop",
						"message": map[string]any{
							"role":    "assistant",
							"content": `<think>Let's call add: <tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call></think>`,
						},
					}},
				})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-add",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":              "assistant",
						"reasoning_content": "We need to add a=2 and b=3 using the add tool.",
						"content":           "",
						"tool_calls": []any{map[string]any{
							"id":       "call_add_1",
							"type":     "function",
							"function": map[string]any{"name": "add", "arguments": `{"a":2,"b":3}`},
						}},
					},
				}},
			})
			return
		}

		// 7) parallel_tool_calls
		if strings.Contains(lastUser, "BOTH Tokyo and Paris") {
			id2 := "call_par_2"
			if faults.duplicateParallelIDs {
				id2 = "call_par_1"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-par",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":    "assistant",
						"content": "",
						"tool_calls": []any{
							map[string]any{"id": "call_par_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Tokyo"}`}},
							map[string]any{"id": id2, "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`}},
						},
					},
				}},
			})
			return
		}

		// 8) single_tool_call
		if len(req.Tools) == 1 && req.Tools[0].Function.Name == "get_weather" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-single",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":    "assistant",
						"content": "",
						"tool_calls": []any{map[string]any{
							"id":       "call_single_1",
							"type":     "function",
							"function": map[string]any{"name": "get_weather", "arguments": `{"city":"Tokyo"}`},
						}},
					},
				}},
			})
			return
		}

		// 9) reasoning_field_compat
		if strings.Contains(lastUser, "bat and a ball") {
			msg := map[string]any{"role": "assistant", "content": "5 cents."}
			switch {
			case faults.reasoningFieldOnly:
				msg["reasoning"] = "Let x be the ball. x + (x + 1.00) = 1.10 => x = 0.05."
			case faults.inlineThinkLeak:
				msg["content"] = "<think>x + (x + 1.00) = 1.10 so x = 0.05.</think>\n5 cents."
			default:
				msg["reasoning_content"] = "Let x be the ball. x + (x + 1.00) = 1.10 => x = 0.05."
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-reason",
				"choices": []any{map[string]any{
					"index":         0,
					"finish_reason": "stop",
					"message":       msg,
				}},
			})
			return
		}

		w.WriteHeader(http.StatusBadRequest)
	}))
}

func findResult(t *testing.T, res *probe.RunResult, id string) probe.ProbeResult {
	t.Helper()
	for _, r := range res.Results {
		if r.ID == id {
			return r
		}
	}
	t.Fatalf("probe result %q not found in run results", id)
	return probe.ProbeResult{}
}

func TestHealthyVLLMServerAll13ProbesPass(t *testing.T) {
	srv := newMockLLMServer(t, mockServerFaults{engineHeader: "vllm"})
	defer srv.Close()

	runner, err := probe.NewRunner(probe.Config{
		Endpoint: srv.URL + "/v1",
		Model:    "Qwen/Qwen3-32B",
		Timeout:  5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := runner.Run(context.Background())
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}

	if len(res.Results) != 13 {
		t.Fatalf("expected 13 probes, got %d", len(res.Results))
	}
	if res.Engine.Engine != probe.EngineVLLM || res.Engine.Version != "0.28.1" {
		t.Fatalf("unexpected engine: %+v", res.Engine)
	}
	if res.Counts.Fail != 0 || res.Counts.Warn != 0 || res.Counts.Pass != 13 {
		for _, r := range res.Results {
			t.Logf("%s: %s (%s) - %s", r.ID, r.Status, r.Cause, r.Summary)
		}
		t.Fatalf("expected 13 PASS, got %+v", res.Counts)
	}
	if res.ExitCode() != 0 {
		t.Fatalf("expected exit code 0, got %d", res.ExitCode())
	}
}

func TestHealthyLlamaServerAll13ProbesPass(t *testing.T) {
	srv := newMockLLMServer(t, mockServerFaults{engineHeader: "llama-server"})
	defer srv.Close()

	runner, err := probe.NewRunner(probe.Config{
		Endpoint: srv.URL,
		Timeout:  5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := runner.Run(context.Background())
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}
	if res.Engine.Engine != probe.EngineLlamaServer || res.Engine.Version != "b5120" {
		t.Fatalf("unexpected engine: %+v", res.Engine)
	}
	if res.Model != "Qwen/Qwen3-32B" {
		t.Fatalf("expected auto-selected model Qwen/Qwen3-32B, got %q", res.Model)
	}
	if res.Counts.Pass != 13 {
		t.Fatalf("expected 13 PASS on llama-server, got %+v", res.Counts)
	}
}

func TestRealWorldFailureModes(t *testing.T) {
	t.Run("vLLM_combined_real_world_bugs", func(t *testing.T) {
		srv := newMockLLMServer(t, mockServerFaults{
			engineHeader:       "vllm",
			schemaDropOnCall2:  true,
			decoyMisfire:       true,
			swallowToolInThink: true,
			streamNegToolIndex: true,
			streamSwallowAngle: true,
			kvCacheDrop:        true,
			toolRoleHTTP400:    true,
			xgrammarJSONError:  true,
		})
		defer srv.Close()

		runner, err := probe.NewRunner(probe.Config{
			Endpoint: srv.URL + "/v1",
			Model:    "Qwen/Qwen3-32B",
			Timeout:  5 * time.Second,
		})
		if err != nil {
			t.Fatal(err)
		}
		res, err := runner.Run(context.Background())
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}

		expect := []struct {
			id     string
			status probe.Status
			cause  probe.CauseTag
			substr string
		}{
			{"strict_nested_schema", probe.StatusFail, probe.CauseSchemaBug, "target.path"},
			{"think_decoy_isolation", probe.StatusFail, probe.CauseServerParser, "delete_all"},
			{"tool_swallowed_in_think", probe.StatusFail, probe.CauseServerParser, "swallowed"},
			{"stream_tool_delta", probe.StatusFail, probe.CauseServerParser, "-1"},
			{"stream_angle_bracket", probe.StatusFail, probe.CauseServerParser, "swallowed"},
			{"multi_tool_kv_prefix", probe.StatusFail, probe.CauseKVCacheBreak, "4%"},
			{"tool_role_roundtrip", probe.StatusFail, probe.CauseServerParser, "HTTP 400"},
			{"json_schema_response_format", probe.StatusFail, probe.CauseSchemaBug, "XGrammar"},
		}
		for _, tc := range expect {
			got := findResult(t, res, tc.id)
			if got.Status != tc.status || got.Cause != tc.cause || !strings.Contains(got.Summary, tc.substr) {
				t.Errorf("%s: got status=%s cause=%s summary=%q; want status=%s cause=%s containing %q",
					tc.id, got.Status, got.Cause, got.Summary, tc.status, tc.cause, tc.substr)
			}
		}
		if res.ExitCode() != 1 {
			t.Fatalf("expected exit code 1, got %d", res.ExitCode())
		}
	})

	t.Run("reasoning_field_divergence_only_reasoning", func(t *testing.T) {
		srv := newMockLLMServer(t, mockServerFaults{reasoningFieldOnly: true})
		defer srv.Close()
		// Make decoy/add probes NOT emit reasoning_content so we isolate .reasoning-only cleanly.
		// Build a focused mock specifically for .reasoning-only check.
		focusedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			if r.URL.Path == "/v1/models" {
				_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen3","owned_by":"vllm"}]}`))
				return
			}
			var req client.ChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"1 2 3 4 5"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte(`{"id":"1","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","reasoning":"step 1","content":"5 cents"}}]}`))
		}))
		defer focusedSrv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: focusedSrv.URL, Model: "qwen3", Timeout: 5 * time.Second})
		res, err := runner.Run(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		got := findResult(t, res, "reasoning_field_compat")
		if got.Status != probe.StatusWarn || got.Cause != probe.CauseServerParser || !strings.Contains(got.Summary, ".reasoning") {
			t.Fatalf("expected WARN for .reasoning-only, got %+v", got)
		}
	})

	t.Run("inline_think_leak_in_content", func(t *testing.T) {
		focusedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			if r.URL.Path == "/v1/models" {
				_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen3","owned_by":"llamacpp"}]}`))
				return
			}
			var req client.ChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(`data: {"id":"1","choices":[{"index":0,"delta":{"content":"1 2 3 4 5"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte(`{"id":"1","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"<think>ball is 5c</think>\n5 cents"}}]}`))
		}))
		defer focusedSrv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: focusedSrv.URL, Model: "qwen3", Timeout: 5 * time.Second})
		res, err := runner.Run(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		got := findResult(t, res, "reasoning_field_compat")
		if got.Status != probe.StatusWarn || !strings.Contains(got.Summary, "<think>") {
			t.Fatalf("expected WARN for inline <think>, got %+v", got)
		}
	})

	t.Run("llama_cache_n_drop_and_duplicate_ids_and_markdown_fence_and_missing_done", func(t *testing.T) {
		srv := newMockLLMServer(t, mockServerFaults{
			engineHeader:         "llama-server",
			kvCacheDropLlama:     true,
			duplicateParallelIDs: true,
			jsonMarkdownFence:    true,
			streamMissingDone:    true,
			swallowToolInInline:  true,
		})
		defer srv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: srv.URL, Model: "Qwen/Qwen3-32B", Timeout: 5 * time.Second})
		res, err := runner.Run(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if r := findResult(t, res, "multi_tool_kv_prefix"); r.Status != probe.StatusFail || r.Cause != probe.CauseKVCacheBreak {
			t.Errorf("expected KV_CACHE_BREAK via timings.cache_n, got %+v", r)
		}
		if r := findResult(t, res, "parallel_tool_calls"); r.Status != probe.StatusFail || !strings.Contains(r.Summary, "duplicate") {
			t.Errorf("expected duplicate ID failure, got %+v", r)
		}
		if r := findResult(t, res, "json_schema_response_format"); r.Status != probe.StatusFail || !strings.Contains(r.Summary, "markdown fences") {
			t.Errorf("expected markdown fence failure, got %+v", r)
		}
		if r := findResult(t, res, "stream_tool_delta"); r.Status != probe.StatusWarn || !strings.Contains(r.Summary, "[DONE]") {
			t.Errorf("expected missing [DONE] WARN, got %+v", r)
		}
		if r := findResult(t, res, "tool_swallowed_in_think"); r.Status != probe.StatusFail || r.Cause != probe.CauseServerParser {
			t.Errorf("expected inline <think> swallow FAIL, got %+v", r)
		}
	})

	t.Run("kv_cache_timing_heuristic_fallback", func(t *testing.T) {
		srv := newMockLLMServer(t, mockServerFaults{kvCacheTimingOnly: true})
		defer srv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: srv.URL, Model: "Qwen/Qwen3-32B", Timeout: 5 * time.Second})
		res, err := runner.Run(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		r := findResult(t, res, "multi_tool_kv_prefix")
		if r.Status != probe.StatusFail || r.Cause != probe.CauseKVCacheBreak {
			t.Fatalf("expected timing-based KV_CACHE_BREAK, got %+v", r)
		}
	})

	t.Run("auth_error_returns_exit2_error", func(t *testing.T) {
		srv := newMockLLMServer(t, mockServerFaults{requireAuthKey: "topsecret"})
		defer srv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: srv.URL, Model: "Qwen/Qwen3-32B", APIKey: "bad", Timeout: 5 * time.Second})
		_, err := runner.Run(context.Background())
		var authErr *client.AuthError
		if !errors.As(err, &authErr) {
			t.Fatalf("expected AuthError, got %v", err)
		}
	})

	t.Run("connection_refused_returns_connection_error", func(t *testing.T) {
		// Create and immediately close a server to get a guaranteed refused localhost port.
		srv := httptest.NewServer(http.NotFoundHandler())
		url := srv.URL
		srv.Close()

		runner, _ := probe.NewRunner(probe.Config{Endpoint: url, Model: "test", Timeout: 2 * time.Second})
		_, err := runner.Run(context.Background())
		var connErr *client.ConnectionError
		if !errors.As(err, &connErr) {
			t.Fatalf("expected ConnectionError, got %v", err)
		}
	})
}
