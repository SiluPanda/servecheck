package gguf

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/builtins"
	"github.com/nikolalohinski/gonja/v2/config"
	"github.com/nikolalohinski/gonja/v2/exec"
	"github.com/nikolalohinski/gonja/v2/loaders"

	"github.com/SiluPanda/servecheck/internal/probe"
	"github.com/SiluPanda/servecheck/internal/recipes"
)

func init() {
	// Prevent gonja's internal logrus logger from writing noise to stderr/stdout.
	gonja.SetLoggerOutput(io.Discard)
}

// DoctorReport holds the complete offline GGUF/tokenizer_config + Jinja2 template analysis.
type DoctorReport struct {
	Path         string              `json:"path,omitempty"`
	Format       string              `json:"format"`
	ModelName    string              `json:"model_name,omitempty"`
	Architecture string              `json:"architecture,omitempty"`
	ModelFamily  string              `json:"model_family"`
	TensorCount  uint64              `json:"tensor_count,omitempty"`
	KVCount      uint64              `json:"metadata_kv_count,omitempty"`
	BOSToken     string              `json:"bos_token,omitempty"`
	EOSToken     string              `json:"eos_token,omitempty"`
	TemplateLen  int                 `json:"template_bytes"`
	ElapsedMs    int64               `json:"elapsed_ms"`
	Results      []probe.ProbeResult `json:"results"`
	Counts       probe.SummaryCounts `json:"counts"`
}

// ExitCode returns 1 if any doctor check failed, otherwise 0.
func (r *DoctorReport) ExitCode() int {
	if r.Counts.Fail > 0 {
		return 1
	}
	return 0
}

// AnalyzeFile reads a .gguf or tokenizer_config.json file and runs the full offline Jinja2 doctor.
func AnalyzeFile(path string) (*DoctorReport, error) {
	start := time.Now()
	meta, err := LoadFile(path)
	if err != nil {
		return nil, err
	}
	rep := AnalyzeMetadata(meta)
	rep.ElapsedMs = time.Since(start).Milliseconds()
	return rep, nil
}

// AnalyzeMetadata runs all offline template doctor checks on parsed GGUF or tokenizer_config metadata.
func AnalyzeMetadata(meta *Metadata) *DoctorReport {
	start := time.Now()
	rep := &DoctorReport{
		Path:         meta.Path,
		Format:       meta.Format,
		ModelName:    meta.Name,
		Architecture: meta.Architecture,
		ModelFamily:  recipes.InferModelFamily(meta.Name, meta.Architecture),
		TensorCount:  meta.TensorCount,
		KVCount:      meta.MetadataKVCount,
		BOSToken:     meta.BOSToken,
		EOSToken:     meta.EOSToken,
		TemplateLen:  len(meta.ChatTemplate),
	}

	// Check 1: template_present
	if strings.TrimSpace(meta.ChatTemplate) == "" {
		rep.Results = append(rep.Results, probe.ProbeResult{
			ID:         "template_present",
			Status:     probe.StatusFail,
			Cause:      probe.CauseUnsupported,
			Summary:    "no tokenizer.chat_template found in file metadata",
			DurationMs: time.Since(start).Milliseconds(),
		})
		rep.Counts = probe.CountResults(rep.Results)
		rep.ElapsedMs = time.Since(start).Milliseconds()
		return rep
	}
	rep.Results = append(rep.Results, probe.ProbeResult{
		ID:         "template_present",
		Status:     probe.StatusPass,
		Summary:    fmt.Sprintf("extracted chat_template (%d bytes)", len(meta.ChatTemplate)),
		DurationMs: time.Since(start).Milliseconds(),
	})

	// Check 2: template_syntax_parse
	tParseStart := time.Now()
	tmpl, obs, err := compileHFTemplate(meta.ChatTemplate, time.Date(2026, 11, 21, 10, 0, 0, 0, time.UTC))
	if err != nil {
		rep.Results = append(rep.Results, probe.ProbeResult{
			ID:         "template_syntax_parse",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    "Jinja2 syntax parse failed",
			Details:    err.Error(),
			DurationMs: time.Since(tParseStart).Milliseconds(),
		})
		rep.Counts = probe.CountResults(rep.Results)
		rep.ElapsedMs = time.Since(start).Milliseconds()
		return rep
	}
	rep.Results = append(rep.Results, probe.ProbeResult{
		ID:         "template_syntax_parse",
		Status:     probe.StatusPass,
		Summary:    "Jinja2 chat_template parsed cleanly",
		DurationMs: time.Since(tParseStart).Milliseconds(),
	})

	bos := meta.BOSToken
	eos := meta.EOSToken

	// Check 3: template_tool_render
	toolRes, renderedTurn2 := checkToolRender(tmpl, bos, eos)
	rep.Results = append(rep.Results, toolRes)

	// Check 4: template_prefix_stability (Turn 1 -> Turn 2 strict byte-prefix stability)
	rep.Results = append(rep.Results, checkPrefixStability(meta.ChatTemplate, bos, eos))

	// Check 5: template_dynamic_time (strftime_now / now() detection)
	rep.Results = append(rep.Results, checkDynamicTime(meta.ChatTemplate, bos, eos, obs))

	// Check 6: template_think_tool_hygiene
	rep.Results = append(rep.Results, checkThinkToolHygiene(meta.ChatTemplate, renderedTurn2, bos, eos))

	rep.Counts = probe.CountResults(rep.Results)
	rep.ElapsedMs = time.Since(start).Milliseconds()
	return rep
}

type templateObserver struct {
	usedDynamicTime bool
	timeFormats     []string
	nowTime         time.Time
}

func compileHFTemplate(src string, clock time.Time) (*exec.Template, *templateObserver, error) {
	obs := &templateObserver{nowTime: clock}

	cfg := config.New()
	cfg.TrimBlocks = true
	cfg.LeftStripBlocks = true

	filters := exec.NewFilterSet(map[string]exec.FilterFunction{})
	filters.Update(builtins.Filters)
	_ = filters.Replace("tojson", hfFilterToJSON)
	_ = filters.Register("items", hfFilterItems)
	_ = filters.Register("strip", hfFilterStrip)
	_ = filters.Register("lstrip", hfFilterLStrip)
	_ = filters.Register("rstrip", hfFilterRStrip)

	ctx := builtins.GlobalFunctions.Inherit()
	ctx.Update(builtins.GlobalVariables)
	ctx.Set("raise_exception", func(_ *exec.Evaluator, params *exec.VarArgs) (any, error) {
		msg := "template raised exception"
		if len(params.Args) > 0 {
			msg = params.Args[0].String()
		}
		return nil, errors.New(msg)
	})
	ctx.Set("strftime_now", func(_ *exec.Evaluator, params *exec.VarArgs) (any, error) {
		fmtStr := "%Y-%m-%d"
		if len(params.Args) > 0 {
			fmtStr = params.Args[0].String()
		}
		obs.usedDynamicTime = true
		obs.timeFormats = append(obs.timeFormats, fmtStr)
		return formatStrftime(obs.nowTime, fmtStr), nil
	})
	ctx.Set("now", func(_ *exec.Evaluator, _ *exec.VarArgs) (any, error) {
		obs.usedDynamicTime = true
		return map[string]any{
			"strftime": func(_ *exec.Evaluator, p *exec.VarArgs) (any, error) {
				fmtStr := "%Y-%m-%d"
				if len(p.Args) > 0 {
					fmtStr = p.Args[0].String()
				}
				obs.timeFormats = append(obs.timeFormats, fmtStr)
				return formatStrftime(obs.nowTime, fmtStr), nil
			},
			"isoformat": func(_ *exec.Evaluator, _ *exec.VarArgs) (any, error) {
				obs.timeFormats = append(obs.timeFormats, "%Y-%m-%dT%H:%M:%S")
				return obs.nowTime.Format(time.RFC3339), nil
			},
			"year":   obs.nowTime.Year(),
			"month":  int(obs.nowTime.Month()),
			"day":    obs.nowTime.Day(),
			"hour":   obs.nowTime.Hour(),
			"minute": obs.nowTime.Minute(),
			"second": obs.nowTime.Second(),
		}, nil
	})

	env := &exec.Environment{
		Context:           ctx,
		Filters:           filters,
		Tests:             builtins.Tests,
		ControlStructures: builtins.ControlStructures,
		Methods:           builtins.Methods,
	}

	loader, err := loaders.NewMemoryLoader(map[string]string{
		"/chat_template": src,
	})
	if err != nil {
		return nil, obs, err
	}
	tmpl, err := exec.NewTemplate("/chat_template", cfg, loader, env)
	if err != nil {
		return nil, obs, err
	}
	return tmpl, obs, nil
}

func renderTemplate(tmpl *exec.Template, messages []any, tools []any, bos, eos string, addGenPrompt bool) (string, error) {
	data := exec.NewContext(map[string]any{
		"messages":              messages,
		"tools":                 tools,
		"bos_token":             bos,
		"eos_token":             eos,
		"add_generation_prompt": addGenPrompt,
		"enable_thinking":       true,
	})
	return tmpl.ExecuteToString(data)
}

func sampleTools() []any {
	return []any{
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "Get current weather for a city.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string", "description": "City name"},
						"unit": map[string]any{"type": "string", "enum": []any{"celsius", "fahrenheit"}},
					},
					"required": []any{"city"},
				},
			},
		},
		map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "get_time",
				"description": "Get current local time for a timezone.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"timezone": map[string]any{"type": "string"},
					},
					"required": []any{"timezone"},
				},
			},
		},
	}
}

func sampleMultiTurnMessages(includeTurn2 bool, inlineThink bool) []any {
	assistantContent := ""
	assistantReasoning := ""
	if inlineThink {
		assistantContent = "<think>I should fetch both weather and local time for Tokyo.</think>\n"
	} else {
		assistantReasoning = "I should fetch both weather and local time for Tokyo."
	}

	assistantTurn1 := map[string]any{
		"role":    "assistant",
		"content": assistantContent,
		"tool_calls": []any{
			map[string]any{
				"id":   "call_wx_1",
				"type": "function",
				"function": map[string]any{
					"name": "get_weather",
					"arguments": map[string]any{
						"city": "Tokyo",
						"unit": "celsius",
					},
				},
			},
			map[string]any{
				"id":   "call_tz_2",
				"type": "function",
				"function": map[string]any{
					"name": "get_time",
					"arguments": map[string]any{
						"timezone": "Asia/Tokyo",
					},
				},
			},
		},
	}
	if assistantReasoning != "" {
		assistantTurn1["reasoning_content"] = assistantReasoning
	}

	msgs := []any{
		map[string]any{
			"role":    "system",
			"content": "You are a helpful coding assistant.",
		},
		map[string]any{
			"role":    "user",
			"content": "Check both the weather and local time in Tokyo.",
		},
		assistantTurn1,
		map[string]any{
			"role":         "tool",
			"tool_call_id": "call_wx_1",
			"name":         "get_weather",
			"content":      `{"city":"Tokyo","temp_c":22,"condition":"sunny"}`,
		},
		map[string]any{
			"role":         "tool",
			"tool_call_id": "call_tz_2",
			"name":         "get_time",
			"content":      `{"timezone":"Asia/Tokyo","time":"14:30"}`,
		},
	}

	if includeTurn2 {
		assistantTurn2 := map[string]any{
			"role":    "assistant",
			"content": "It is 22C and sunny in Tokyo, and the local time is 14:30.",
		}
		if inlineThink {
			assistantTurn2["content"] = "<think>Both tool calls succeeded.</think>\nIt is 22C and sunny in Tokyo, and the local time is 14:30."
		} else {
			assistantTurn2["reasoning_content"] = "Both tool calls succeeded."
		}
		msgs = append(msgs,
			assistantTurn2,
			map[string]any{
				"role":    "user",
				"content": "Thanks! Now check Osaka too.",
			},
		)
	}
	return msgs
}

// Check 3: template_tool_render
func checkToolRender(tmpl *exec.Template, bos, eos string) (probe.ProbeResult, string) {
	start := time.Now()
	tools := sampleTools()
	msgs := sampleMultiTurnMessages(false, true)

	rendered, err := renderTemplate(tmpl, msgs, tools, bos, eos, true)
	dur := time.Since(start).Milliseconds()
	if err != nil {
		return probe.ProbeResult{
			ID:         "template_tool_render",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    "template failed to render tools + role=\"tool\" conversation",
			Details:    err.Error(),
			DurationMs: dur,
		}, ""
	}

	hasToolDef := strings.Contains(rendered, "get_weather") && strings.Contains(rendered, "get_time")
	hasToolResp := strings.Contains(rendered, "sunny") && strings.Contains(rendered, "14:30")
	if !hasToolDef {
		return probe.ProbeResult{
			ID:         "template_tool_render",
			Status:     probe.StatusFail,
			Cause:      probe.CauseUnsupported,
			Summary:    "template rendered without error but silently dropped tool definitions",
			DurationMs: dur,
		}, rendered
	}
	if !hasToolResp {
		return probe.ProbeResult{
			ID:         "template_tool_render",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    "template dropped role=\"tool\" response content from rendered prompt",
			DurationMs: dur,
		}, rendered
	}

	return probe.ProbeResult{
		ID:         "template_tool_render",
		Status:     probe.StatusPass,
		Summary:    fmt.Sprintf("rendered tool definitions + multi-tool calls + role=\"tool\" cleanly (%d bytes)", len(rendered)),
		DurationMs: dur,
	}, rendered
}

// Check 4: template_prefix_stability
func checkPrefixStability(tmplSrc, bos, eos string) probe.ProbeResult {
	start := time.Now()
	fixedClock := time.Date(2026, 11, 21, 10, 0, 0, 0, time.UTC)
	tmpl, _, err := compileHFTemplate(tmplSrc, fixedClock)
	if err != nil {
		return probe.ProbeResult{
			ID:         "template_prefix_stability",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    "template failed to compile for prefix stability check",
			Details:    err.Error(),
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	tools := sampleTools()

	// Test BOTH inline <think> (in assistant.content) and structured reasoning_content modes!
	modes := []struct {
		label       string
		inlineThink bool
	}{
		{label: "inline-<think>", inlineThink: true},
		{label: "reasoning_content", inlineThink: false},
	}

	for _, m := range modes {
		// Step A: [system, user1] -> [system, user1, assistant1, tool1, tool2]
		turn1Base := sampleMultiTurnMessages(false, m.inlineThink)[:2]
		turn1AfterTools := sampleMultiTurnMessages(false, m.inlineThink)
		turn2Full := sampleMultiTurnMessages(true, m.inlineThink)

		rBaseNoGen, err1 := renderTemplate(tmpl, turn1Base, tools, bos, eos, false)
		rTurn1NoGen, err2 := renderTemplate(tmpl, turn1AfterTools, tools, bos, eos, false)
		rTurn1WithGen, err3 := renderTemplate(tmpl, turn1AfterTools, tools, bos, eos, true)
		rTurn2WithGen, err4 := renderTemplate(tmpl, turn2Full, tools, bos, eos, true)
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			firstErr := firstNonNil(err1, err2, err3, err4)
			return probe.ProbeResult{
				ID:         "template_prefix_stability",
				Status:     probe.StatusFail,
				Cause:      probe.CauseServerParser,
				Summary:    fmt.Sprintf("template errored during multi-turn rendering (%s)", m.label),
				Details:    firstErr.Error(),
				DurationMs: time.Since(start).Milliseconds(),
			}
		}

		// Verify Step A prefix: [system, user1] must remain a strict prefix once assistant+tools are appended.
		if !strings.HasPrefix(rTurn1WithGen, rBaseNoGen) {
			idx := firstDiffIndex(rBaseNoGen, rTurn1WithGen)
			return probe.ProbeResult{
				ID:         "template_prefix_stability",
				Status:     probe.StatusFail,
				Cause:      probe.CauseKVCacheBreak,
				Summary:    fmt.Sprintf("Turn 1 system/user prefix mutated at byte %d after appending tool turn — busts KV cache", idx),
				Details:    formatDiffContext(rBaseNoGen, rTurn1WithGen, idx),
				DurationMs: time.Since(start).Milliseconds(),
				Metrics:    map[string]any{"first_diff_byte": idx, "mode": m.label},
			}
		}

		// Verify Step B prefix: Turn 1 (with <think> + 2 tool calls + tool results) must remain
		// a strict byte-for-byte prefix of Turn 2 when the next user message arrives!
		if !strings.HasPrefix(rTurn2WithGen, rTurn1NoGen) {
			idx := firstDiffIndex(rTurn1NoGen, rTurn2WithGen)
			reason := diagnosePrefixBreak(rTurn1NoGen, rTurn2WithGen, idx)
			return probe.ProbeResult{
				ID:         "template_prefix_stability",
				Status:     probe.StatusFail,
				Cause:      probe.CauseKVCacheBreak,
				Summary:    fmt.Sprintf("Turn 2 mutated Turn 1 prefix at byte %d (%s) — busts llama.cpp/vLLM KV cache!", idx, reason),
				Details:    formatDiffContext(rTurn1NoGen, rTurn2WithGen, idx),
				DurationMs: time.Since(start).Milliseconds(),
				Metrics:    map[string]any{"first_diff_byte": idx, "mode": m.label, "reason": reason},
			}
		}
	}

	return probe.ProbeResult{
		ID:         "template_prefix_stability",
		Status:     probe.StatusPass,
		Summary:    "Turn 1 -> Turn 2 strict byte-prefix stability preserved (inline <think>, reasoning_content, & multi-tool)",
		DurationMs: time.Since(start).Milliseconds(),
	}
}

func diagnosePrefixBreak(turn1, turn2 string, idx int) string {
	t1Tail := turn1[max(0, idx-32):min(len(turn1), idx+96)]
	t2Tail := turn2[max(0, idx-32):min(len(turn2), idx+96)]
	t1HasThink := strings.Contains(strings.ToLower(turn1), "<think>") && strings.Contains(turn1, "fetch both weather")
	t2HasThink := strings.Contains(turn2[:min(len(turn2), len(turn1))], "fetch both weather")
	if t1HasThink && !t2HasThink {
		return "retroactively stripped Turn 1 <think> block"
	}
	if strings.Contains(t1Tail, "<think>") && !strings.Contains(t2Tail, "<think>") {
		return "retroactively altered/stripped Turn 1 <think> tags"
	}
	if strings.Contains(t1Tail, "get_weather") || strings.Contains(t1Tail, "Tokyo") || strings.Contains(t1Tail, "tool_call") {
		return "mutated Turn 1 tool_call serialization"
	}
	return "prior-turn bytes shifted on Turn 2"
}

// Check 5: template_dynamic_time
var dynamicTimeStaticRe = regexp.MustCompile(`(?i)\b(strftime_now|now\s*\(|datetime\.now|datetime\.today|current_date)\b`)

func checkDynamicTime(tmplSrc, bos, eos string, initialObs *templateObserver) probe.ProbeResult {
	start := time.Now()
	t1 := time.Date(2026, 11, 21, 10, 0, 0, 0, time.UTC)
	t1PlusMin := time.Date(2026, 11, 21, 10, 1, 30, 0, time.UTC)
	t2NextDay := time.Date(2026, 11, 22, 10, 0, 0, 0, time.UTC)

	tools := sampleTools()
	msgs := sampleMultiTurnMessages(false, true)

	renderAt := func(clock time.Time) (string, *templateObserver, error) {
		tm, ob, err := compileHFTemplate(tmplSrc, clock)
		if err != nil {
			return "", ob, err
		}
		out, err := renderTemplate(tm, msgs, tools, bos, eos, true)
		return out, ob, err
	}

	r1, ob1, err1 := renderAt(t1)
	r1m, _, err2 := renderAt(t1PlusMin)
	r2d, _, err3 := renderAt(t2NextDay)

	dur := time.Since(start).Milliseconds()
	if err1 == nil && err2 == nil && r1 != r1m {
		idx := firstDiffIndex(r1, r1m)
		return probe.ProbeResult{
			ID:         "template_dynamic_time",
			Status:     probe.StatusFail,
			Cause:      probe.CauseKVCacheBreak,
			Summary:    fmt.Sprintf("live clock/timestamp injected at byte %d (changes every minute/second) — busts KV cache on every turn", idx),
			Details:    formatDiffContext(r1, r1m, idx),
			DurationMs: dur,
		}
	}
	if err1 == nil && err3 == nil && r1 != r2d {
		idx := firstDiffIndex(r1, r2d)
		return probe.ProbeResult{
			ID:         "template_dynamic_time",
			Status:     probe.StatusWarn,
			Cause:      probe.CauseKVCacheBreak,
			Summary:    fmt.Sprintf("current date injected via strftime_now/now() at byte %d — invalidates KV cache across midnight", idx),
			Details:    formatDiffContext(r1, r2d, idx),
			DurationMs: dur,
		}
	}

	usedDyn := (initialObs != nil && initialObs.usedDynamicTime) || (ob1 != nil && ob1.usedDynamicTime)
	if usedDyn || dynamicTimeStaticRe.MatchString(tmplSrc) {
		return probe.ProbeResult{
			ID:         "template_dynamic_time",
			Status:     probe.StatusWarn,
			Cause:      probe.CauseKVCacheBreak,
			Summary:    "template references dynamic date/time helper (strftime_now/now())",
			DurationMs: dur,
		}
	}

	return probe.ProbeResult{
		ID:         "template_dynamic_time",
		Status:     probe.StatusPass,
		Summary:    "no dynamic date/time injection detected (prompt prefix is time-invariant)",
		DurationMs: dur,
	}
}

// Check 6: template_think_tool_hygiene
func checkThinkToolHygiene(tmplSrc, renderedTurn2, bos, eos string) probe.ProbeResult {
	start := time.Now()
	if renderedTurn2 == "" {
		return probe.ProbeResult{
			ID:         "template_think_tool_hygiene",
			Status:     probe.StatusWarn,
			Cause:      probe.CauseUnsupported,
			Summary:    "skipped hygiene check because tool render did not produce output",
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	// Check for doubled <think><think> (happens when template blindly wraps message.content that already contains <think>).
	if strings.Contains(renderedTurn2, "<think><think>") || strings.Contains(renderedTurn2, "<think>\n<think>") {
		return probe.ProbeResult{
			ID:         "template_think_tool_hygiene",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    "template double-wraps existing <think> blocks (<think><think>)",
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	// Check balanced <tool_call> and </tool_call> tags when used.
	openTool := strings.Count(renderedTurn2, "<tool_call>")
	closeTool := strings.Count(renderedTurn2, "</tool_call>")
	if openTool != closeTool {
		return probe.ProbeResult{
			ID:         "template_think_tool_hygiene",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    fmt.Sprintf("unbalanced <tool_call> tags in rendered prompt (open=%d, close=%d)", openTool, closeTool),
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	// Check balanced <think> and </think> tags, allowing one intentional trailing <think> at generation prompt end.
	openThink := strings.Count(renderedTurn2, "<think>")
	closeThink := strings.Count(renderedTurn2, "</think>")
	trailingThink := strings.HasSuffix(strings.TrimSpace(renderedTurn2), "<think>")
	if openThink != closeThink && !(openThink == closeThink+1 && trailingThink) {
		return probe.ProbeResult{
			ID:         "template_think_tool_hygiene",
			Status:     probe.StatusFail,
			Cause:      probe.CauseServerParser,
			Summary:    fmt.Sprintf("unbalanced <think> tags in rendered prompt (open=%d, close=%d)", openThink, closeThink),
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	// Check if template strips thinking conditionally via last_query_index pattern (static advisory if dynamic didn't trigger).
	hasLastQueryStrip := strings.Contains(tmplSrc, "last_query_index") && strings.Contains(tmplSrc, "</think>")
	if hasLastQueryStrip {
		return probe.ProbeResult{
			ID:         "template_think_tool_hygiene",
			Status:     probe.StatusWarn,
			Cause:      probe.CauseKVCacheBreak,
			Summary:    "template contains last_query_index + </think> logic (risk of retroactive thinking strip on multi-turn)",
			DurationMs: time.Since(start).Milliseconds(),
		}
	}

	note := "balanced <think> and <tool_call> tags; no double-wrapping"
	if trailingThink {
		note += " (auto-opens <think> at generation prompt)"
	}
	return probe.ProbeResult{
		ID:         "template_think_tool_hygiene",
		Status:     probe.StatusPass,
		Summary:    note,
		DurationMs: time.Since(start).Milliseconds(),
	}
}

// --- HuggingFace-compatible Jinja2 filters & helpers ---

func hfFilterToJSON(_ *exec.Evaluator, in *exec.Value, params *exec.VarArgs) *exec.Value {
	if in.IsError() {
		return in
	}

	var indent int
	hasIndent := false
	sortKeys := false
	ensureASCII := false

	if v, ok := params.KwArgs["indent"]; ok && !v.IsNil() && v.IsInteger() {
		indent = v.Integer()
		hasIndent = true
	} else if len(params.Args) > 0 && !params.Args[0].IsNil() && params.Args[0].IsInteger() {
		indent = params.Args[0].Integer()
		hasIndent = true
	}
	if v, ok := params.KwArgs["sort_keys"]; ok {
		sortKeys = v.Bool()
	}
	if v, ok := params.KwArgs["ensure_ascii"]; ok {
		ensureASCII = v.Bool()
	}
	// Accept "separators" kwarg silently for Python json.dumps parity.
	_ = sortKeys

	val := unwrapGonjaValue(in.Interface())
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if hasIndent && indent > 0 {
		enc.SetIndent("", strings.Repeat(" ", indent))
	}
	if err := enc.Encode(val); err != nil {
		return exec.ValueError(fmt.Errorf("tojson marshal error: %w", err))
	}
	out := strings.TrimSuffix(buf.String(), "\n")
	if ensureASCII {
		out = escapeNonASCII(out)
	}
	return exec.AsSafeValue(out)
}

func unwrapGonjaValue(v any) any {
	switch x := v.(type) {
	case nil:
		return nil
	case *exec.Value:
		if x.IsNil() {
			return nil
		}
		return unwrapGonjaValue(x.Interface())
	case *exec.Dict:
		m := make(map[string]any, len(x.Pairs))
		for _, p := range x.Pairs {
			m[p.Key.String()] = unwrapGonjaValue(p.Value)
		}
		return m
	case exec.Dict:
		m := make(map[string]any, len(x.Pairs))
		for _, p := range x.Pairs {
			m[p.Key.String()] = unwrapGonjaValue(p.Value)
		}
		return m
	case map[string]any:
		m := make(map[string]any, len(x))
		for k, val := range x {
			m[k] = unwrapGonjaValue(val)
		}
		return m
	case []any:
		s := make([]any, len(x))
		for i, val := range x {
			s[i] = unwrapGonjaValue(val)
		}
		return s
	default:
		return x
	}
}

func escapeNonASCII(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r > 0x7f {
			fmt.Fprintf(&b, `\u%04x`, r)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func hfFilterItems(_ *exec.Evaluator, in *exec.Value, _ *exec.VarArgs) *exec.Value {
	if in.IsError() {
		return in
	}
	switch x := in.Interface().(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]any, 0, len(keys))
		for _, k := range keys {
			out = append(out, []any{k, x[k]})
		}
		return exec.AsValue(out)
	case *exec.Dict:
		out := make([]any, 0, len(x.Pairs))
		for _, p := range x.Pairs {
			out = append(out, []any{p.Key.Interface(), p.Value.Interface()})
		}
		return exec.AsValue(out)
	default:
		return exec.AsValue([]any{})
	}
}

func hfFilterStrip(_ *exec.Evaluator, in *exec.Value, params *exec.VarArgs) *exec.Value {
	if in.IsError() {
		return in
	}
	s := in.String()
	if len(params.Args) > 0 {
		return exec.AsValue(strings.Trim(s, params.Args[0].String()))
	}
	return exec.AsValue(strings.TrimSpace(s))
}

func hfFilterLStrip(_ *exec.Evaluator, in *exec.Value, params *exec.VarArgs) *exec.Value {
	if in.IsError() {
		return in
	}
	s := in.String()
	if len(params.Args) > 0 {
		return exec.AsValue(strings.TrimLeft(s, params.Args[0].String()))
	}
	return exec.AsValue(strings.TrimLeft(s, " \t\n\r\f\v"))
}

func hfFilterRStrip(_ *exec.Evaluator, in *exec.Value, params *exec.VarArgs) *exec.Value {
	if in.IsError() {
		return in
	}
	s := in.String()
	if len(params.Args) > 0 {
		return exec.AsValue(strings.TrimRight(s, params.Args[0].String()))
	}
	return exec.AsValue(strings.TrimRight(s, " \t\n\r\f\v"))
}

func formatStrftime(t time.Time, format string) string {
	repl := strings.NewReplacer(
		"%%", "%",
		"%Y", t.Format("2006"),
		"%y", t.Format("06"),
		"%m", t.Format("01"),
		"%d", t.Format("02"),
		"%e", fmt.Sprintf("%2d", t.Day()),
		"%B", t.Format("January"),
		"%b", t.Format("Jan"),
		"%A", t.Format("Monday"),
		"%a", t.Format("Mon"),
		"%H", t.Format("15"),
		"%I", t.Format("03"),
		"%M", t.Format("04"),
		"%S", t.Format("05"),
		"%p", t.Format("PM"),
		"%Z", t.Format("MST"),
		"%z", t.Format("-0700"),
		"%F", t.Format("2006-01-02"),
		"%T", t.Format("15:04:05"),
	)
	return repl.Replace(format)
}

func firstDiffIndex(a, b string) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func formatDiffContext(a, b string, idx int) string {
	start := max(0, idx-24)
	aEnd := min(len(a), idx+56)
	bEnd := min(len(b), idx+56)
	aSeg := strings.ReplaceAll(a[start:aEnd], "\n", `\n`)
	bSeg := strings.ReplaceAll(b[start:bEnd], "\n", `\n`)
	return fmt.Sprintf("Turn1: %q vs Turn2: %q", aSeg, bSeg)
}

func firstNonNil(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
