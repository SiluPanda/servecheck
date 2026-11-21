package probe

import "time"

// Status represents the outcome of a single diagnostic probe or offline check.
type Status string

const (
	StatusPass Status = "PASS"
	StatusWarn Status = "WARN"
	StatusFail Status = "FAIL"
)

// CauseTag classifies the root cause of a WARN or FAIL diagnostic finding.
type CauseTag string

const (
	CauseNone           CauseTag = ""
	CauseServerParser   CauseTag = "SERVER_PARSER_BUG"
	CauseKVCacheBreak   CauseTag = "KV_CACHE_BREAK"
	CauseSchemaBug      CauseTag = "SCHEMA_BUG"
	CauseModelFloorFail CauseTag = "MODEL_FLOOR_FAIL"
	CauseUnsupported    CauseTag = "UNSUPPORTED"
)

// Supported local LLM inference engine identifiers.
const (
	EngineAuto        = "auto"
	EngineVLLM        = "vllm"
	EngineLlamaServer = "llama-server"
	EngineSGLang      = "sglang"
	EngineOllama      = "ollama"
	EngineMLX         = "mlx"
	EngineUnknown     = "unknown"
)

// EngineInfo describes the detected or user-specified inference engine.
type EngineInfo struct {
	Engine       string `json:"engine"`
	Version      string `json:"version,omitempty"`
	DetectedFrom string `json:"detected_from,omitempty"`
	ServerHeader string `json:"server_header,omitempty"`
}

// Display returns a human-readable engine label such as "vLLM 0.28.1".
func (e EngineInfo) Display() string {
	name := DisplayEngineName(e.Engine)
	if e.Version != "" {
		return name + " " + e.Version
	}
	return name
}

// DisplayEngineName returns the canonical display name for an engine slug.
func DisplayEngineName(engine string) string {
	switch engine {
	case EngineVLLM:
		return "vLLM"
	case EngineLlamaServer:
		return "llama-server"
	case EngineSGLang:
		return "SGLang"
	case EngineOllama:
		return "Ollama"
	case EngineMLX:
		return "mlx-lm"
	default:
		if engine == "" || engine == EngineUnknown || engine == EngineAuto {
			return "OpenAI-compat (unknown engine)"
		}
		return engine
	}
}

// ProbeResult captures the outcome and evidence of one probe execution.
type ProbeResult struct {
	ID         string         `json:"id"`
	Status     Status         `json:"status"`
	Cause      CauseTag       `json:"cause,omitempty"`
	Summary    string         `json:"summary"`
	Details    string         `json:"details,omitempty"`
	DurationMs int64          `json:"duration_ms"`
	Metrics    map[string]any `json:"metrics,omitempty"`
}

// SummaryCounts aggregates probe statuses.
type SummaryCounts struct {
	Pass  int `json:"pass"`
	Warn  int `json:"warn"`
	Fail  int `json:"fail"`
	Total int `json:"total"`
}

// CountResults computes pass/warn/fail totals from a slice of ProbeResults.
func CountResults(results []ProbeResult) SummaryCounts {
	var c SummaryCounts
	for _, r := range results {
		c.Total++
		switch r.Status {
		case StatusPass:
			c.Pass++
		case StatusWarn:
			c.Warn++
		case StatusFail:
			c.Fail++
		}
	}
	return c
}

// RunResult is the complete output of running the server conformance probe suite.
type RunResult struct {
	Endpoint    string        `json:"endpoint"`
	Model       string        `json:"model"`
	ModelFamily string        `json:"model_family"`
	Engine      EngineInfo    `json:"engine"`
	ElapsedMs   int64         `json:"elapsed_ms"`
	Timestamp   time.Time     `json:"timestamp"`
	Results     []ProbeResult `json:"results"`
	Counts      SummaryCounts `json:"counts"`
}

// ExitCode returns 1 if any probe failed, otherwise 0.
func (r *RunResult) ExitCode() int {
	if r.Counts.Fail > 0 {
		return 1
	}
	return 0
}
