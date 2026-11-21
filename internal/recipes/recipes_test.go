package recipes

import (
	"strings"
	"testing"
)

func TestLoadEmbeddedRecipes(t *testing.T) {
	all, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if len(all) < 20 {
		t.Fatalf("expected >= 20 recipes, got %d", len(all))
	}

	seenIDs := map[string]bool{}
	engines := map[string]bool{}
	for _, r := range all {
		if r.ID == "" || r.Engine == "" || r.Flags == "" || len(r.Probes) == 0 {
			t.Errorf("incomplete recipe: %+v", r)
		}
		if seenIDs[r.ID] {
			t.Errorf("duplicate recipe ID: %s", r.ID)
		}
		seenIDs[r.ID] = true
		engines[r.Engine] = true
	}

	for _, reqEngine := range []string{"vllm", "llama-server", "sglang", "ollama", "mlx"} {
		if !engines[reqEngine] {
			t.Errorf("missing recipes for engine %q", reqEngine)
		}
	}
}

func TestInferModelFamily(t *testing.T) {
	cases := []struct {
		model, arch, want string
	}{
		{"Qwen/Qwen3-32B", "", "qwen3"},
		{"", "qwen3moe", "qwen3"},
		{"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "", "deepseek"},
		{"deepseek-ai/DeepSeek-V3", "", "deepseek"},
		{"meta-llama/Llama-3.3-70B-Instruct", "", "llama"},
		{"google/gemma-3-27b-it", "", "gemma"},
		{"mistralai/Devstral-Small", "", "mistral"},
		{"custom/unknown-model", "", "*"},
	}
	for _, tc := range cases {
		got := InferModelFamily(tc.model, tc.arch)
		if got != tc.want {
			t.Errorf("InferModelFamily(%q, %q) = %q, want %q", tc.model, tc.arch, got, tc.want)
		}
	}
}

func TestMatchPrefersFamilySpecificAndDeduplicates(t *testing.T) {
	fixes, err := Match("vllm", "qwen3", []string{
		"strict_nested_schema",
		"stream_angle_bracket",
		"reasoning_field_compat",
		"multi_tool_kv_prefix",
	})
	if err != nil {
		t.Fatalf("Match() error: %v", err)
	}
	if len(fixes) < 3 {
		t.Fatalf("expected >= 3 fixes, got %d: %+v", len(fixes), fixes)
	}
	foundQwenTool := false
	foundQwenReason := false
	foundKV := false
	for _, f := range fixes {
		if f.RecipeID == "vllm-qwen3-tool-strict" {
			foundQwenTool = true
			if !strings.Contains(f.Flags, "--tool-strict-level parameter") || !strings.Contains(f.Flags, "qwen3_coder") {
				t.Errorf("unexpected flags: %s", f.Flags)
			}
			if len(f.MatchedProbes) != 2 {
				t.Errorf("expected 2 merged probes for vllm-qwen3-tool-strict, got %v", f.MatchedProbes)
			}
		}
		if f.RecipeID == "vllm-qwen3-reasoning" {
			foundQwenReason = true
		}
		if f.RecipeID == "vllm-kv-prefix-cache" {
			foundKV = true
		}
	}
	if !foundQwenTool || !foundQwenReason || !foundKV {
		t.Fatalf("missing expected recipes: qwenTool=%v qwenReason=%v kv=%v (fixes=%+v)", foundQwenTool, foundQwenReason, foundKV, fixes)
	}
}

func TestMatchUnknownEngineSuggestsVLLMAndLlamaServer(t *testing.T) {
	fixes, err := Match("unknown", "qwen3", []string{"multi_tool_kv_prefix"})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, f := range fixes {
		seen[f.Engine] = true
	}
	if !seen["vllm"] || !seen["llama-server"] {
		t.Fatalf("expected fixes for both vllm and llama-server when engine unknown, got %+v", fixes)
	}
}
