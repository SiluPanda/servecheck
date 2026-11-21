package report

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/servecheck/servecheck/internal/gguf"
	"github.com/servecheck/servecheck/internal/probe"
	"github.com/servecheck/servecheck/internal/recipes"
)

func sampleRunResult() *probe.RunResult {
	results := []probe.ProbeResult{
		{ID: "models_endpoint", Status: probe.StatusPass, Summary: `model "Qwen/Qwen3-32B" advertised`, DurationMs: 5},
		{ID: "basic_chat", Status: probe.StatusPass, Summary: "OK (TTFT 42ms, 86.4 tok/s)", DurationMs: 120},
		{ID: "strict_nested_schema", Status: probe.StatusFail, Cause: probe.CauseSchemaBug, Summary: `1/3 calls dropped required field "target.path"`, Details: "call 2 missing target.path", DurationMs: 310},
		{ID: "reasoning_field_compat", Status: probe.StatusWarn, Cause: probe.CauseServerParser, Summary: "emits .reasoning without .reasoning_content", DurationMs: 95},
	}
	return &probe.RunResult{
		Endpoint:    "http://localhost:8000/v1",
		Model:       "Qwen/Qwen3-32B",
		ModelFamily: "qwen3",
		Engine:      probe.EngineInfo{Engine: probe.EngineVLLM, Version: "0.28.1", DetectedFrom: "header:x-vllm-version"},
		ElapsedMs:   530,
		Timestamp:   time.Date(2026, 11, 21, 6, 0, 0, 0, time.UTC),
		Results:     results,
		Counts:      probe.CountResults(results),
	}
}

func TestFormatServerReportAllFormats(t *testing.T) {
	res := sampleRunResult()
	fixes, err := recipes.Match("vllm", "qwen3", []string{"strict_nested_schema", "reasoning_field_compat"})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("table_no_color", func(t *testing.T) {
		var buf bytes.Buffer
		if err := FormatServerReport(&buf, res, fixes, Options{Format: FormatTable, Color: false, Version: "0.1.0"}); err != nil {
			t.Fatal(err)
		}
		out := buf.String()
		if strings.Contains(out, "\033[") {
			t.Errorf("unexpected ANSI escapes when Color=false:\n%s", out)
		}
		for _, want := range []string{"vLLM 0.28.1", "[PASS]", "[WARN]", "[FAIL]", "[SCHEMA_BUG]", "--tool-strict-level parameter", "--reasoning-parser qwen3", "exit 1"} {
			if !strings.Contains(out, want) {
				t.Errorf("table output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("table_with_color", func(t *testing.T) {
		var buf bytes.Buffer
		if err := FormatServerReport(&buf, res, fixes, Options{Format: FormatTable, Color: true, Version: "0.1.0"}); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(buf.String(), "\033[") {
			t.Errorf("expected ANSI escapes when Color=true")
		}
	})

	t.Run("markdown", func(t *testing.T) {
		var buf bytes.Buffer
		if err := FormatServerReport(&buf, res, fixes, Options{Format: FormatMarkdown, Version: "0.1.0"}); err != nil {
			t.Fatal(err)
		}
		out := buf.String()
		for _, want := range []string{"## servecheck Report", "| Status | Check | Cause | Summary | Time |", "```bash", "--tool-strict-level parameter"} {
			if !strings.Contains(out, want) {
				t.Errorf("markdown output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("json", func(t *testing.T) {
		var buf bytes.Buffer
		if err := FormatServerReport(&buf, res, fixes, Options{Format: FormatJSON, Version: "0.1.0"}); err != nil {
			t.Fatal(err)
		}
		var parsed map[string]any
		if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
			t.Fatalf("invalid JSON output: %v\n%s", err, buf.String())
		}
		if parsed["model"] != "Qwen/Qwen3-32B" {
			t.Errorf("unexpected JSON model: %v", parsed["model"])
		}
		if _, ok := parsed["recommended_fixes"]; !ok {
			t.Errorf("missing recommended_fixes in JSON")
		}
	})
}

func TestFormatGGUFAndRecipes(t *testing.T) {
	rep := &gguf.DoctorReport{
		Path:         "./qwen3.gguf",
		Format:       "gguf-v3",
		ModelName:    "Qwen3-32B",
		Architecture: "qwen3",
		ModelFamily:  "qwen3",
		TensorCount:  256,
		KVCount:      32,
		ElapsedMs:    12,
		Results: []probe.ProbeResult{
			{ID: "template_prefix_stability", Status: probe.StatusFail, Cause: probe.CauseKVCacheBreak, Summary: "Turn 2 stripped <think>", DurationMs: 4},
		},
		Counts: probe.SummaryCounts{Fail: 1, Total: 1},
	}
	fixes, _ := recipes.Match("llama-server", "qwen3", []string{"template_prefix_stability"})

	for _, f := range []Format{FormatTable, FormatMarkdown, FormatJSON} {
		var buf bytes.Buffer
		if err := FormatGGUFReport(&buf, rep, fixes, Options{Format: f, Version: "0.1.0"}); err != nil {
			t.Fatalf("FormatGGUFReport(%s) error: %v", f, err)
		}
		if !strings.Contains(buf.String(), "template_prefix_stability") {
			t.Errorf("FormatGGUFReport(%s) missing check ID", f)
		}
	}

	allRecipes, _ := recipes.Filter("vllm", "")
	for _, f := range []Format{FormatTable, FormatMarkdown, FormatJSON} {
		var buf bytes.Buffer
		if err := FormatRecipes(&buf, allRecipes, Options{Format: f}); err != nil {
			t.Fatalf("FormatRecipes(%s) error: %v", f, err)
		}
		if !strings.Contains(buf.String(), "qwen3_coder") {
			t.Errorf("FormatRecipes(%s) missing qwen3_coder", f)
		}
	}
}
