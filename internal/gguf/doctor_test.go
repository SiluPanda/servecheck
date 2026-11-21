package gguf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/SiluPanda/servecheck/internal/probe"
)

// Realistic Qwen3 template that retroactively strips Turn 1 <think> blocks using ns.last_query_index
// (the notorious llama.cpp / vLLM KV-cache-busting template pattern!).
const qwen3KVBustingTemplate = `{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == 'system' %}{{ messages[0].content }}{% else %}You are a helpful assistant.{% endif %}
{{- '\n\n# Tools\n\n<tools>\n' }}
{%- for tool in tools %}
{{- tool | tojson(ensure_ascii=False) }}
{{- '\n' }}
{%- endfor %}
{{- '</tools><|im_end|>\n' }}
{%- else %}
{%- if messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- endif %}
{%- set ns = namespace(last_query_index=messages|length - 1) %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {%- set ns.last_query_index = loop.index0 %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>assistant\n' }}
        {%- if loop.index0 > ns.last_query_index and reasoning_content %}
            {{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- else %}
            {{- content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\n<tool_call>\n{"name": "' + tool_call.name + '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson(ensure_ascii=False) }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}`

// Prefix-stable Qwen3 template: keeps prior-turn <think> blocks intact across turns so KV cache stays hot!
const qwen3PrefixStableTemplate = `{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == 'system' %}{{ messages[0].content }}{% else %}You are a helpful assistant.{% endif %}
{{- '\n\n# Tools\n\n<tools>\n' }}
{%- for tool in tools %}
{{- tool | tojson(ensure_ascii=False) }}
{{- '\n' }}
{%- endfor %}
{{- '</tools><|im_end|>\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' }}
        {%- if message.reasoning_content is defined and message.reasoning_content %}
            {{- '<think>\n' + message.reasoning_content.strip('\n') + '\n</think>\n' }}
        {%- endif %}
        {{- message.content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- set fn = tool_call.function %}
                {{- '\n<tool_call>\n{"name": "' + fn.name + '", "arguments": ' + (fn.arguments | tojson(ensure_ascii=False)) + '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}`

// Realistic DeepSeek-R1/V3 prefix-stable template with <｜User｜>/<｜Assistant｜> and auto-opened <think>.
const deepseekPrefixStableTemplate = `{{ bos_token }}{%- if messages[0].role == 'system' %}{{ messages[0].content }}{% endif -%}
{%- if tools %}\nTools:\n{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}{% endif -%}
{%- for message in messages -%}
{%- if message.role == 'user' %}<｜User｜>{{ message.content }}{% elif message.role == 'assistant' %}<｜Assistant｜>{% if message.reasoning_content is defined and message.reasoning_content %}<think>{{ message.reasoning_content }}</think>{% endif %}{{ message.content }}{% if message.tool_calls %}{% for tc in message.tool_calls %}<tool_call>{{ tc.function | tojson }}</tool_call>{% endfor %}{% endif %}{{ eos_token }}{% elif message.role == 'tool' %}<tool_response>{{ message.content }}</tool_response>{% endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<｜Assistant｜><think>{% endif %}`

// Realistic Llama 3.1/3.3 template with strftime_now("%d %b %Y") date injection.
const llama31DateInjectionTemplate = `{{- bos_token }}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}
{{- 'Cutting Knowledge Date: December 2023\n' }}
{{- 'Today Date: ' + strftime_now('%d %b %Y') + '\n\n' }}
{%- if messages[0].role == 'system' %}{{ messages[0].content }}{% endif %}
{%- if tools %}
{{- '\n\nAvailable functions:\n' }}
{%- for t in tools %}{{ t | tojson }}{{ '\n' }}{% endfor %}
{%- endif %}
{{- '<|eot_id|>' }}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message.content + '<|eot_id|>' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message.content }}
        {%- if message.tool_calls %}
            {%- for tc in message.tool_calls %}
                {{- '{"name": "' + tc.function.name + '", "parameters": ' + (tc.function.arguments | tojson) + '}' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|eot_id|>' }}
    {%- elif message.role == 'tool' %}
        {{- '<|start_header_id|>ipython<|end_header_id|>\n\n' + message.content + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}`

// Template injecting live HH:MM:SS timestamp via strftime_now (busts KV cache every single second!).
const liveClockKVBustingTemplate = `{{- '<|im_start|>system\nCurrent time: ' + strftime_now('%Y-%m-%d %H:%M:%S') + '\n' }}
{%- for t in tools %}{{ t | tojson }}{% endfor %}<|im_end|>\n
{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}`

// Realistic Gemma template that raises an exception on role="tool"!
const gemmaNoToolRoleTemplate = `{{ bos_token }}{%- for message in messages -%}
{%- if message.role not in ['user', 'assistant', 'model', 'system'] -%}
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
{%- endif -%}
<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n
{%- endfor -%}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}`

// Buggy template that double-wraps inline <think> blocks.
const doubleThinkBugTemplate = `{%- for t in tools %}{{ t | tojson }}{% endfor %}
{%- for m in messages -%}
{%- if m.role == 'assistant' %}<|im_start|>assistant\n<think>{{ m.content }}</think><|im_end|>\n{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif -%}
{%- endfor -%}`

func findCheck(t *testing.T, rep *DoctorReport, id string) probe.ProbeResult {
	t.Helper()
	for _, r := range rep.Results {
		if r.ID == id {
			return r
		}
	}
	t.Fatalf("doctor check %q not found", id)
	return probe.ProbeResult{}
}

func TestDoctorQwen3KVBustingVsPrefixStableGGUF(t *testing.T) {
	dir := t.TempDir()

	// 1. Synthetic GGUF v3 with KV-cache-busting Qwen3 template
	badBytes := buildSyntheticGGUF(t, 3, 256, []ggufKVItem{
		{"general.architecture", TypeString, "qwen3"},
		{"general.name", TypeString, "Qwen3-35B-A3B-Q4_K_M"},
		{"tokenizer.ggml.tokens", TypeArray, ggufArrayVal{elemType: TypeString, elems: []any{"<|im_start|>", "<|im_end|>"}}},
		{"tokenizer.ggml.bos_token_id", TypeUint32, uint32(0)},
		{"tokenizer.ggml.eos_token_id", TypeUint32, uint32(1)},
		{"tokenizer.chat_template", TypeString, qwen3KVBustingTemplate},
	}, 256*1024)
	badPath := filepath.Join(dir, "qwen3-bad.gguf")
	if err := os.WriteFile(badPath, badBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	badRep, err := AnalyzeFile(badPath)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("AnalyzeFile(bad) error: %v", err)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("offline doctor took %v (expected fast <40ms typical)", elapsed)
	}
	if badRep.ExitCode() != 1 {
		t.Fatalf("expected exit 1 for KV-busting Qwen3 template, got %d (%+v)", badRep.ExitCode(), badRep.Results)
	}
	prefixCheck := findCheck(t, badRep, "template_prefix_stability")
	if prefixCheck.Status != probe.StatusFail || prefixCheck.Cause != probe.CauseKVCacheBreak || !strings.Contains(prefixCheck.Summary, "<think>") {
		t.Fatalf("expected KV_CACHE_BREAK mentioning <think>, got %+v", prefixCheck)
	}
	toolCheck := findCheck(t, badRep, "template_tool_render")
	if toolCheck.Status != probe.StatusPass {
		t.Fatalf("expected template_tool_render PASS, got %+v", toolCheck)
	}

	// 2. Synthetic GGUF v3 with prefix-stable fixed Qwen3 template
	goodBytes := buildSyntheticGGUF(t, 3, 256, []ggufKVItem{
		{"general.architecture", TypeString, "qwen3"},
		{"general.name", TypeString, "Qwen3-35B-A3B-Fixed"},
		{"tokenizer.chat_template", TypeString, qwen3PrefixStableTemplate},
	}, 128*1024)
	goodPath := filepath.Join(dir, "qwen3-good.gguf")
	if err := os.WriteFile(goodPath, goodBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	goodRep, err := AnalyzeFile(goodPath)
	if err != nil {
		t.Fatalf("AnalyzeFile(good) error: %v", err)
	}
	if goodRep.ExitCode() != 0 || goodRep.Counts.Fail != 0 || goodRep.Counts.Warn != 0 {
		for _, r := range goodRep.Results {
			t.Logf("%s: %s (%s) - %s [%s]", r.ID, r.Status, r.Cause, r.Summary, r.Details)
		}
		t.Fatalf("expected all PASS for prefix-stable Qwen3 template, got %+v", goodRep.Counts)
	}
}

func TestDoctorDeepSeekLlamaGemmaAndEdgeCases(t *testing.T) {
	t.Run("deepseek_prefix_stable_auto_think", func(t *testing.T) {
		rep := AnalyzeMetadata(&Metadata{
			Format:       "gguf-v3",
			Name:         "DeepSeek-R1-Distill",
			Architecture: "deepseek2",
			BOSToken:     "<｜begin▁of▁sentence｜>",
			EOSToken:     "<｜end▁of▁sentence｜>",
			ChatTemplate: deepseekPrefixStableTemplate,
		})
		if rep.ExitCode() != 0 || rep.Counts.Fail != 0 {
			for _, r := range rep.Results {
				t.Logf("%s: %s - %s (%s)", r.ID, r.Status, r.Summary, r.Details)
			}
			t.Fatalf("expected DeepSeek prefix-stable template to pass, got %+v", rep.Counts)
		}
		hyg := findCheck(t, rep, "template_think_tool_hygiene")
		if !strings.Contains(hyg.Summary, "auto-opens <think>") {
			t.Errorf("expected auto-opens <think> note, got %q", hyg.Summary)
		}
	})

	t.Run("llama31_date_injection_warns_not_fails", func(t *testing.T) {
		rep := AnalyzeMetadata(&Metadata{
			Format:       "tokenizer_config.json",
			Name:         "Meta-Llama-3.1-8B-Instruct",
			Architecture: "llama",
			BOSToken:     "<|begin_of_text|>",
			EOSToken:     "<|eot_id|>",
			ChatTemplate: llama31DateInjectionTemplate,
		})
		if rep.Counts.Fail != 0 {
			for _, r := range rep.Results {
				t.Logf("%s: %s - %s (%s)", r.ID, r.Status, r.Summary, r.Details)
			}
			t.Fatalf("expected 0 FAIL for Llama 3.1 date template, got %+v", rep.Counts)
		}
		dt := findCheck(t, rep, "template_dynamic_time")
		if dt.Status != probe.StatusWarn || dt.Cause != probe.CauseKVCacheBreak {
			t.Fatalf("expected WARN KV_CACHE_BREAK for day-level strftime_now, got %+v", dt)
		}
	})

	t.Run("live_clock_timestamp_fails_kv_cache", func(t *testing.T) {
		rep := AnalyzeMetadata(&Metadata{
			Format:       "gguf-v3",
			Name:         "ClockBroken-7B",
			ChatTemplate: liveClockKVBustingTemplate,
		})
		dt := findCheck(t, rep, "template_dynamic_time")
		if dt.Status != probe.StatusFail || dt.Cause != probe.CauseKVCacheBreak {
			t.Fatalf("expected FAIL KV_CACHE_BREAK for live HH:MM:SS clock, got %+v", dt)
		}
	})

	t.Run("now_strftime_method_detected", func(t *testing.T) {
		nowTmpl := `{{- '<|im_start|>system\nNow: ' + now().strftime('%Y-%m-%d %H:%M:%S') + '\n' }}
{%- for t in tools %}{{ t | tojson }}{% endfor %}<|im_end|>\n
{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}`
		rep := AnalyzeMetadata(&Metadata{
			Format:       "tokenizer_config.json",
			Name:         "NowClock-7B",
			ChatTemplate: nowTmpl,
		})
		dt := findCheck(t, rep, "template_dynamic_time")
		if dt.Status != probe.StatusFail || dt.Cause != probe.CauseKVCacheBreak {
			t.Fatalf("expected FAIL KV_CACHE_BREAK for now().strftime, got %+v", dt)
		}
	})

	t.Run("gemma_rejects_tool_role", func(t *testing.T) {
		rep := AnalyzeMetadata(&Metadata{
			Format:       "gguf-v3",
			Name:         "gemma-2-9b-it",
			Architecture: "gemma2",
			BOSToken:     "<bos>",
			ChatTemplate: gemmaNoToolRoleTemplate,
		})
		tr := findCheck(t, rep, "template_tool_render")
		if tr.Status != probe.StatusFail || tr.Cause != probe.CauseServerParser || !strings.Contains(tr.Details, "Conversation roles must alternate") {
			t.Fatalf("expected Gemma tool role FAIL with raise_exception message, got %+v", tr)
		}
	})

	t.Run("double_wrapped_think_hygiene_fail", func(t *testing.T) {
		rep := AnalyzeMetadata(&Metadata{
			Format:       "gguf-v3",
			Name:         "broken-think",
			ChatTemplate: doubleThinkBugTemplate,
		})
		hyg := findCheck(t, rep, "template_think_tool_hygiene")
		if hyg.Status != probe.StatusFail || !strings.Contains(hyg.Summary, "double-wraps") {
			t.Fatalf("expected double-wrap FAIL, got %+v", hyg)
		}
	})

	t.Run("syntax_error_and_missing_template", func(t *testing.T) {
		syn := AnalyzeMetadata(&Metadata{Format: "gguf-v3", ChatTemplate: "{% if unclosed %}"})
		if syn.ExitCode() != 1 || findCheck(t, syn, "template_syntax_parse").Status != probe.StatusFail {
			t.Fatalf("expected syntax error FAIL, got %+v", syn.Results)
		}

		miss := AnalyzeMetadata(&Metadata{Format: "gguf-v3", ChatTemplate: ""})
		if miss.ExitCode() != 1 || findCheck(t, miss, "template_present").Status != probe.StatusFail {
			t.Fatalf("expected missing template FAIL, got %+v", miss.Results)
		}
	})
}
