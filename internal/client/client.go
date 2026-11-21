package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// EngineMeta describes the detected inference engine and version from HTTP metadata.
type EngineMeta struct {
	Engine       string
	Version      string
	DetectedFrom string
	ServerHeader string
}

// ConnectionError indicates a network/transport failure reaching the endpoint (exit code 2).
type ConnectionError struct {
	URL string
	Err error
}

func (e *ConnectionError) Error() string {
	return fmt.Sprintf("connection failed for %s: %v", e.URL, e.Err)
}

func (e *ConnectionError) Unwrap() error { return e.Err }

// AuthError indicates HTTP 401/403 authentication failure (exit code 2).
type AuthError struct {
	StatusCode int
	Body       string
}

func (e *AuthError) Error() string {
	return fmt.Sprintf("authentication failed (HTTP %d): pass --api-key or set OPENAI_API_KEY", e.StatusCode)
}

// Client talks to an OpenAI-compatible inference server.
type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	Timeout    time.Duration
}

// New constructs a Client with normalized BaseURL and timeout.
func New(endpoint, apiKey string, timeout time.Duration) (*Client, error) {
	base, err := NormalizeBaseURL(endpoint)
	if err != nil {
		return nil, err
	}
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   min(timeout, 10*time.Second),
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          10,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   min(timeout, 10*time.Second),
		ExpectContinueTimeout: 1 * time.Second,
	}
	return &Client{
		BaseURL: base,
		APIKey:  apiKey,
		Timeout: timeout,
		HTTPClient: &http.Client{
			Transport: transport,
		},
	}, nil
}

// NormalizeBaseURL trims trailing slashes and appends /v1 when no path is given.
func NormalizeBaseURL(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", errors.New("endpoint URL is required")
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint URL %q: %w", raw, err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("invalid endpoint URL %q: missing host", raw)
	}
	p := strings.TrimRight(u.Path, "/")
	if p == "" {
		p = "/v1"
	}
	u.Path = p
	return strings.TrimRight(u.String(), "/"), nil
}

// ModelsResponse models GET /v1/models.
type ModelsResponse struct {
	Object     string        `json:"object"`
	Data       []ModelItem   `json:"data"`
	HTTPStatus int           `json:"-"`
	Headers    http.Header   `json:"-"`
	RawBody    []byte        `json:"-"`
	Elapsed    time.Duration `json:"-"`
}

// ModelItem represents one entry in /v1/models.
type ModelItem struct {
	ID          string         `json:"id"`
	Object      string         `json:"object,omitempty"`
	OwnedBy     string         `json:"owned_by,omitempty"`
	Root        string         `json:"root,omitempty"`
	MaxModelLen int            `json:"max_model_len,omitempty"`
	Meta        map[string]any `json:"meta,omitempty"`
}

// ChatRequest is an OpenAI /v1/chat/completions request payload.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// Message represents a chat message in requests and responses.
type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	Reasoning        string     `json:"reasoning,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`

	// Presence flags populated when decoding responses so empty vs absent fields are distinguishable.
	HasReasoning        bool `json:"-"`
	HasReasoningContent bool `json:"-"`
}

// UnmarshalJSON decodes Message while recording exact presence of reasoning fields.
func (m *Message) UnmarshalJSON(data []byte) error {
	type wireMessage struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		Reasoning        *string         `json:"reasoning"`
		ReasoningContent *string         `json:"reasoning_content"`
		ToolCalls        []ToolCall      `json:"tool_calls"`
		ToolCallID       string          `json:"tool_call_id"`
		Name             string          `json:"name"`
	}
	var w wireMessage
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	m.Role = w.Role
	m.ToolCalls = w.ToolCalls
	m.ToolCallID = w.ToolCallID
	m.Name = w.Name

	if len(w.Content) > 0 && string(w.Content) != "null" {
		var s string
		if err := json.Unmarshal(w.Content, &s); err == nil {
			m.Content = s
		} else {
			// Handle multimodal/array content blocks cleanly if returned.
			var parts []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err2 := json.Unmarshal(w.Content, &parts); err2 == nil {
				var b strings.Builder
				for _, p := range parts {
					b.WriteString(p.Text)
				}
				m.Content = b.String()
			}
		}
	}
	if w.Reasoning != nil {
		m.HasReasoning = true
		m.Reasoning = *w.Reasoning
	}
	if w.ReasoningContent != nil {
		m.HasReasoningContent = true
		m.ReasoningContent = *w.ReasoningContent
	}
	return nil
}

// Tool describes a callable function tool.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction defines a tool's name, description, and JSON Schema parameters.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

// ToolCall represents an assistant tool invocation (both non-streaming and streaming delta).
type ToolCall struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction carries the function name and JSON-encoded arguments string.
type ToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ResponseFormat models OpenAI structured outputs response_format.
type ResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}

// JSONSchemaFormat configures type="json_schema" constrained decoding.
type JSONSchemaFormat struct {
	Name   string         `json:"name"`
	Strict bool           `json:"strict"`
	Schema map[string]any `json:"schema"`
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage captures OpenAI usage plus prompt-cache hit extensions.
type Usage struct {
	PromptTokens         int                  `json:"prompt_tokens"`
	CompletionTokens     int                  `json:"completion_tokens"`
	TotalTokens          int                  `json:"total_tokens"`
	PromptTokensDetails  *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	PromptCacheHitTokens *int                 `json:"prompt_cache_hit_tokens,omitempty"`
}

// PromptTokensDetails holds OpenAI/vLLM cached prompt token telemetry.
type PromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens,omitempty"`
}

// Timings captures llama.cpp / llama-server timing and KV-cache metrics.
type Timings struct {
	CacheN             *int    `json:"cache_n,omitempty"`
	PromptN            int     `json:"prompt_n,omitempty"`
	PromptMs           float64 `json:"prompt_ms,omitempty"`
	PredictedN         int     `json:"predicted_n,omitempty"`
	PredictedMs        float64 `json:"predicted_ms,omitempty"`
	PredictedPerSecond float64 `json:"predicted_per_second,omitempty"`
}

// ChatResponse is the parsed non-streaming /v1/chat/completions response.
type ChatResponse struct {
	ID         string        `json:"id"`
	Object     string        `json:"object"`
	Model      string        `json:"model"`
	Choices    []Choice      `json:"choices"`
	Usage      *Usage        `json:"usage,omitempty"`
	Timings    *Timings      `json:"timings,omitempty"`
	HTTPStatus int           `json:"-"`
	Headers    http.Header   `json:"-"`
	RawBody    []byte        `json:"-"`
	Elapsed    time.Duration `json:"-"`
}

// CachedTokens extracts the best available cached prompt token count and its source.
func (r *ChatResponse) CachedTokens() (tokens int, source string, ok bool) {
	if r == nil {
		return 0, "", false
	}
	if r.Usage != nil && r.Usage.PromptTokensDetails != nil && r.Usage.PromptTokensDetails.CachedTokens != nil {
		return *r.Usage.PromptTokensDetails.CachedTokens, "usage.prompt_tokens_details.cached_tokens", true
	}
	if r.Usage != nil && r.Usage.PromptCacheHitTokens != nil {
		return *r.Usage.PromptCacheHitTokens, "usage.prompt_cache_hit_tokens", true
	}
	if r.Timings != nil && r.Timings.CacheN != nil {
		return *r.Timings.CacheN, "timings.cache_n", true
	}
	return 0, "", false
}

// PromptTokens returns the prompt token count from usage or llama.cpp timings.
func (r *ChatResponse) PromptTokens() int {
	if r == nil {
		return 0
	}
	if r.Usage != nil && r.Usage.PromptTokens > 0 {
		return r.Usage.PromptTokens
	}
	if r.Timings != nil {
		total := r.Timings.PromptN
		if r.Timings.CacheN != nil {
			total += *r.Timings.CacheN
		}
		return total
	}
	return 0
}

// StreamChunk represents a single parsed SSE data JSON object.
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
	Timings *Timings       `json:"timings,omitempty"`
}

// StreamChoice represents one choice delta inside an SSE chunk.
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// StreamResult captures an entire SSE /v1/chat/completions stream.
type StreamResult struct {
	HTTPStatus  int
	Headers     http.Header
	ContentType string
	Chunks      []StreamChunk
	SawDone     bool
	ParseErrors []string
	RawBody     []byte
	TTFT        time.Duration
	Elapsed     time.Duration
	// NonStreamFallback is populated if a server ignored stream:true and returned application/json.
	NonStreamFallback *ChatResponse
}

func (c *Client) applyHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "servecheck/0.1")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
}

// Models executes GET <BaseURL>/models.
func (c *Client) Models(ctx context.Context) (*ModelsResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	endpoint := c.BaseURL + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	c.applyHeaders(req)

	start := time.Now()
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, &ConnectionError{URL: endpoint, Err: err}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	elapsed := time.Since(start)
	if err != nil {
		return nil, &ConnectionError{URL: endpoint, Err: err}
	}

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, &AuthError{StatusCode: resp.StatusCode, Body: truncate(string(body), 300)}
	}

	out := &ModelsResponse{
		HTTPStatus: resp.StatusCode,
		Headers:    resp.Header.Clone(),
		RawBody:    body,
		Elapsed:    elapsed,
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		_ = json.Unmarshal(body, out)
	}
	return out, nil
}

// ChatCompletion executes a non-streaming POST <BaseURL>/chat/completions.
func (c *Client) ChatCompletion(ctx context.Context, payload ChatRequest) (*ChatResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	payload.Stream = false
	reqBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	endpoint := c.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyHeaders(req)

	start := time.Now()
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, &ConnectionError{URL: endpoint, Err: err}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	elapsed := time.Since(start)
	if err != nil {
		return nil, &ConnectionError{URL: endpoint, Err: err}
	}

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, &AuthError{StatusCode: resp.StatusCode, Body: truncate(string(body), 300)}
	}

	out := &ChatResponse{
		HTTPStatus: resp.StatusCode,
		Headers:    resp.Header.Clone(),
		RawBody:    body,
		Elapsed:    elapsed,
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if err := json.Unmarshal(body, out); err != nil {
			return out, fmt.Errorf("invalid JSON in chat/completions response: %w", err)
		}
	}
	return out, nil
}

// ChatCompletionStream executes a streaming POST <BaseURL>/chat/completions and collects SSE chunks.
func (c *Client) ChatCompletionStream(ctx context.Context, payload ChatRequest) (*StreamResult, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	payload.Stream = true
	reqBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	endpoint := c.BaseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	c.applyHeaders(req)

	start := time.Now()
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, &ConnectionError{URL: endpoint, Err: err}
	}
	defer resp.Body.Close()

	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	res := &StreamResult{
		HTTPStatus:  resp.StatusCode,
		Headers:     resp.Header.Clone(),
		ContentType: ct,
	}

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, &AuthError{StatusCode: resp.StatusCode, Body: truncate(string(body), 300)}
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		res.RawBody = body
		res.Elapsed = time.Since(start)
		return res, nil
	}

	// Handle servers that ignore stream:true and reply with application/json.
	if strings.Contains(ct, "application/json") {
		body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		res.Elapsed = time.Since(start)
		res.TTFT = res.Elapsed
		res.RawBody = body
		if err != nil {
			return nil, &ConnectionError{URL: endpoint, Err: err}
		}
		var fallback ChatResponse
		fallback.HTTPStatus = resp.StatusCode
		fallback.Headers = resp.Header.Clone()
		fallback.RawBody = body
		fallback.Elapsed = res.Elapsed
		if err := json.Unmarshal(body, &fallback); err == nil {
			res.NonStreamFallback = &fallback
		} else {
			res.ParseErrors = append(res.ParseErrors, err.Error())
		}
		return res, nil
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var rawBuf bytes.Buffer
	for scanner.Scan() {
		line := scanner.Text()
		if rawBuf.Len() < 64*1024 {
			rawBuf.WriteString(line)
			rawBuf.WriteByte('\n')
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, ":") || strings.HasPrefix(trimmed, "event:") || strings.HasPrefix(trimmed, "id:") {
			continue
		}
		if !strings.HasPrefix(trimmed, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			res.SawDone = true
			continue
		}
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			res.ParseErrors = append(res.ParseErrors, fmt.Sprintf("invalid SSE JSON %q: %v", truncate(data, 80), err))
			continue
		}
		if res.TTFT == 0 && chunkHasToken(&chunk) {
			res.TTFT = time.Since(start)
		}
		res.Chunks = append(res.Chunks, chunk)
	}
	if err := scanner.Err(); err != nil {
		res.ParseErrors = append(res.ParseErrors, fmt.Sprintf("stream read error: %v", err))
	}
	res.RawBody = rawBuf.Bytes()
	res.Elapsed = time.Since(start)
	if res.TTFT == 0 {
		res.TTFT = res.Elapsed
	}
	return res, nil
}

func chunkHasToken(c *StreamChunk) bool {
	for _, ch := range c.Choices {
		if ch.Delta.Content != "" || ch.Delta.Reasoning != "" || ch.Delta.ReasoningContent != "" || len(ch.Delta.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

var versionRe = regexp.MustCompile(`(?i)(?:vllm|llama\.cpp|llama-server|sglang|ollama|mlx[-_]?lm)[/:\s]+v?([a-z0-9][a-z0-9.\-_+]*)`)

// DetectEngine identifies the local inference engine and version from HTTP headers, /models payload, or user override.
func DetectEngine(headers http.Header, models *ModelsResponse, override string) EngineMeta {
	ov := strings.ToLower(strings.TrimSpace(override))
	if ov != "" && ov != "auto" {
		return EngineMeta{
			Engine:       normalizeEngineOverride(ov),
			DetectedFrom: "flag:--engine",
			ServerHeader: headerVal(headers, "Server"),
		}
	}

	serverHdr := headerVal(headers, "Server")
	serverLow := strings.ToLower(serverHdr)

	// 1. Explicit engine headers take highest priority.
	if v := headerVal(headers, "X-Vllm-Version"); v != "" {
		return EngineMeta{Engine: "vllm", Version: strings.TrimPrefix(v, "v"), DetectedFrom: "header:x-vllm-version", ServerHeader: serverHdr}
	}
	for k := range headers {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "x-vllm-") {
			return EngineMeta{Engine: "vllm", Version: extractVersion(serverHdr), DetectedFrom: "header:" + lk, ServerHeader: serverHdr}
		}
		if strings.HasPrefix(lk, "x-sglang-") {
			ver := headerVal(headers, "X-Sglang-Version")
			if ver == "" {
				ver = extractVersion(serverHdr)
			}
			return EngineMeta{Engine: "sglang", Version: strings.TrimPrefix(ver, "v"), DetectedFrom: "header:" + lk, ServerHeader: serverHdr}
		}
		if strings.HasPrefix(lk, "x-ollama-") {
			return EngineMeta{Engine: "ollama", Version: extractVersion(serverHdr), DetectedFrom: "header:" + lk, ServerHeader: serverHdr}
		}
	}

	// 2. Server header inspection.
	switch {
	case strings.Contains(serverLow, "vllm"):
		return EngineMeta{Engine: "vllm", Version: extractVersion(serverHdr), DetectedFrom: "header:server", ServerHeader: serverHdr}
	case strings.Contains(serverLow, "llama.cpp") || strings.Contains(serverLow, "llama-server"):
		return EngineMeta{Engine: "llama-server", Version: extractVersion(serverHdr), DetectedFrom: "header:server", ServerHeader: serverHdr}
	case strings.Contains(serverLow, "sglang"):
		return EngineMeta{Engine: "sglang", Version: extractVersion(serverHdr), DetectedFrom: "header:server", ServerHeader: serverHdr}
	case strings.Contains(serverLow, "ollama"):
		return EngineMeta{Engine: "ollama", Version: extractVersion(serverHdr), DetectedFrom: "header:server", ServerHeader: serverHdr}
	case strings.Contains(serverLow, "mlx"):
		return EngineMeta{Engine: "mlx", Version: extractVersion(serverHdr), DetectedFrom: "header:server", ServerHeader: serverHdr}
	}

	// 3. /models payload signatures.
	if models != nil {
		for _, m := range models.Data {
			ob := strings.ToLower(strings.TrimSpace(m.OwnedBy))
			switch {
			case strings.Contains(ob, "vllm"):
				return EngineMeta{Engine: "vllm", DetectedFrom: "/models:owned_by", ServerHeader: serverHdr}
			case strings.Contains(ob, "llamacpp") || strings.Contains(ob, "llama.cpp"):
				return EngineMeta{Engine: "llama-server", DetectedFrom: "/models:owned_by", ServerHeader: serverHdr}
			case strings.Contains(ob, "sglang"):
				return EngineMeta{Engine: "sglang", DetectedFrom: "/models:owned_by", ServerHeader: serverHdr}
			case ob == "library" || strings.Contains(ob, "ollama"):
				return EngineMeta{Engine: "ollama", DetectedFrom: "/models:owned_by", ServerHeader: serverHdr}
			case strings.Contains(ob, "mlx"):
				return EngineMeta{Engine: "mlx", DetectedFrom: "/models:owned_by", ServerHeader: serverHdr}
			}
			if m.MaxModelLen > 0 && strings.Contains(serverLow, "uvicorn") {
				return EngineMeta{Engine: "vllm", DetectedFrom: "/models:max_model_len+uvicorn", ServerHeader: serverHdr}
			}
			if len(m.Meta) > 0 {
				if _, hasVocab := m.Meta["vocab_type"]; hasVocab {
					return EngineMeta{Engine: "llama-server", DetectedFrom: "/models:meta.vocab_type", ServerHeader: serverHdr}
				}
			}
		}
	}

	return EngineMeta{
		Engine:       "unknown",
		DetectedFrom: "auto",
		ServerHeader: serverHdr,
	}
}

func normalizeEngineOverride(s string) string {
	switch s {
	case "llama.cpp", "llamacpp", "llama_server", "llama-server":
		return "llama-server"
	case "mlx-lm", "mlx_lm", "mlx":
		return "mlx"
	case "vllm":
		return "vllm"
	case "sglang":
		return "sglang"
	case "ollama":
		return "ollama"
	default:
		return s
	}
}

func headerVal(h http.Header, key string) string {
	if h == nil {
		return ""
	}
	return strings.TrimSpace(h.Get(key))
}

func extractVersion(serverHeader string) string {
	m := versionRe.FindStringSubmatch(serverHeader)
	if len(m) >= 2 {
		return m[1]
	}
	return ""
}

func truncate(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
