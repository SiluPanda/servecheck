package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/SiluPanda/servecheck/internal/gguf"
	"github.com/SiluPanda/servecheck/internal/probe"
	"github.com/SiluPanda/servecheck/internal/recipes"
	"github.com/SiluPanda/servecheck/internal/report"
)

// version can be overridden at build time via -ldflags "-X main.version=...".
var version = "0.1.0"

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}

	switch args[0] {
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	case "version", "--version", "-v":
		fmt.Fprintf(stdout, "servecheck v%s\n", version)
		return 0
	case "gguf":
		return runGGUF(args[1:], stdout, stderr)
	case "recipes":
		return runRecipes(args[1:], stdout, stderr)
	default:
		return runProbe(args, stdout, stderr)
	}
}

var validEngines = []string{
	probe.EngineAuto,
	probe.EngineVLLM,
	probe.EngineLlamaServer,
	probe.EngineSGLang,
	probe.EngineOllama,
	probe.EngineMLX,
}

func runProbe(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("servecheck", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		model     string
		apiKey    string
		engine    string
		timeout   time.Duration
		formatStr string
		noColor   bool
	)
	fs.StringVar(&model, "model", "", "Model ID to test (auto-selected from /models if omitted)")
	fs.StringVar(&apiKey, "api-key", "", "Bearer API key (or set OPENAI_API_KEY)")
	fs.StringVar(&engine, "engine", probe.EngineAuto, "Target engine: auto|vllm|llama-server|sglang|ollama|mlx")
	fs.DurationVar(&timeout, "timeout", 20*time.Second, "Per-probe HTTP request timeout")
	fs.StringVar(&formatStr, "format", "table", "Output format: table|json|markdown")
	fs.BoolVar(&noColor, "no-color", false, "Disable ANSI terminal colors")
	fs.Usage = func() { printUsage(stderr) }

	// Reorder flags before positional args so both
	// `servecheck http://localhost:8000/v1 --model foo` and `servecheck --model foo http://localhost:8000/v1` work.
	reordered := reorderFlags(args, map[string]bool{
		"--model": true, "-model": true,
		"--api-key": true, "-api-key": true,
		"--engine": true, "-engine": true,
		"--timeout": true, "-timeout": true,
		"--format": true, "-format": true,
	})
	if err := fs.Parse(reordered); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	pos := fs.Args()
	if len(pos) == 0 {
		fmt.Fprintln(stderr, "error: missing <endpoint-url>")
		printUsage(stderr)
		return 2
	}
	if len(pos) > 1 {
		fmt.Fprintf(stderr, "error: unexpected extra argument %q\n", pos[1])
		return 2
	}
	endpoint := pos[0]

	engNorm := normalizeEngineFlag(engine)
	if !slices.Contains(validEngines, engNorm) {
		fmt.Fprintf(stderr, "error: invalid --engine %q (expected auto|vllm|llama-server|sglang|ollama|mlx)\n", engine)
		return 2
	}

	fmtMode, err := report.ParseFormat(formatStr)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}
	if timeout <= 0 {
		fmt.Fprintln(stderr, "error: --timeout must be positive")
		return 2
	}

	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	runner, err := probe.NewRunner(probe.Config{
		Endpoint:       endpoint,
		Model:          model,
		APIKey:         apiKey,
		EngineOverride: engNorm,
		Timeout:        timeout,
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	res, err := runner.Run(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	failedOrWarned := collectIssueProbeIDs(res.Results)
	fixes, err := recipes.Match(res.Engine.Engine, res.ModelFamily, failedOrWarned)
	if err != nil {
		fmt.Fprintf(stderr, "warning: failed to load remediation recipes: %v\n", err)
	}

	opts := report.Options{
		Format:  fmtMode,
		Color:   shouldUseColor(noColor, stdout),
		Version: version,
	}
	if err := report.FormatServerReport(stdout, res, fixes, opts); err != nil {
		fmt.Fprintf(stderr, "error: render report: %v\n", err)
		return 2
	}
	return res.ExitCode()
}

func runGGUF(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("servecheck gguf", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		formatStr string
		engine    string
		noColor   bool
	)
	fs.StringVar(&formatStr, "format", "table", "Output format: table|json|markdown")
	fs.StringVar(&engine, "engine", probe.EngineLlamaServer, "Engine for fix recipes: llama-server|vllm|sglang|ollama|mlx|auto")
	fs.BoolVar(&noColor, "no-color", false, "Disable ANSI terminal colors")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "Usage: servecheck gguf <path-to-.gguf-or-tokenizer_config.json> [--format table|json|markdown] [--engine llama-server|vllm|...] [--no-color]")
	}

	reordered := reorderFlags(args, map[string]bool{
		"--format": true, "-format": true,
		"--engine": true, "-engine": true,
	})
	if err := fs.Parse(reordered); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	pos := fs.Args()
	if len(pos) == 0 {
		fmt.Fprintln(stderr, "error: missing path to .gguf or tokenizer_config.json")
		fs.Usage()
		return 2
	}
	if len(pos) > 1 {
		fmt.Fprintf(stderr, "error: unexpected extra argument %q\n", pos[1])
		return 2
	}

	engNorm := normalizeEngineFlag(engine)
	if !slices.Contains(validEngines, engNorm) {
		fmt.Fprintf(stderr, "error: invalid --engine %q\n", engine)
		return 2
	}

	fmtMode, err := report.ParseFormat(formatStr)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	rep, err := gguf.AnalyzeFile(pos[0])
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	issueIDs := collectIssueProbeIDs(rep.Results)
	fixes, err := recipes.Match(engNorm, rep.ModelFamily, issueIDs)
	if err != nil {
		fmt.Fprintf(stderr, "warning: failed to load remediation recipes: %v\n", err)
	}

	opts := report.Options{
		Format:  fmtMode,
		Color:   shouldUseColor(noColor, stdout),
		Version: version,
	}
	if err := report.FormatGGUFReport(stdout, rep, fixes, opts); err != nil {
		fmt.Fprintf(stderr, "error: render report: %v\n", err)
		return 2
	}
	return rep.ExitCode()
}

func runRecipes(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("servecheck recipes", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		engine      string
		modelFamily string
		formatStr   string
		noColor     bool
	)
	fs.StringVar(&engine, "engine", "all", "Filter by engine: all|vllm|llama-server|sglang|ollama|mlx")
	fs.StringVar(&modelFamily, "model-family", "", "Filter by model family (e.g. qwen3, deepseek, llama, mistral)")
	fs.StringVar(&formatStr, "format", "table", "Output format: table|json|markdown")
	fs.BoolVar(&noColor, "no-color", false, "Disable ANSI terminal colors")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "Usage: servecheck recipes [--engine all|vllm|llama-server|sglang|ollama|mlx] [--model-family <family>] [--format table|json|markdown] [--no-color]")
	}

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	fmtMode, err := report.ParseFormat(formatStr)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	list, err := recipes.Filter(engine, modelFamily)
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}

	opts := report.Options{
		Format:  fmtMode,
		Color:   shouldUseColor(noColor, stdout),
		Version: version,
	}
	if err := report.FormatRecipes(stdout, list, opts); err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 2
	}
	return 0
}

func collectIssueProbeIDs(results []probe.ProbeResult) []string {
	var ids []string
	for _, r := range results {
		if r.Status == probe.StatusFail || r.Status == probe.StatusWarn {
			ids = append(ids, r.ID)
		}
	}
	return ids
}

func normalizeEngineFlag(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "llama.cpp", "llamacpp", "llama_server", "llama-server":
		return probe.EngineLlamaServer
	case "mlx-lm", "mlx_lm", "mlx":
		return probe.EngineMLX
	default:
		return strings.ToLower(strings.TrimSpace(s))
	}
}

// reorderFlags moves flag tokens ahead of positional args while respecting "--" and value-taking flags.
func reorderFlags(args []string, takesValue map[string]bool) []string {
	var flags []string
	var positionals []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			positionals = append(positionals, args[i+1:]...)
			break
		}
		if strings.HasPrefix(a, "-") && a != "-" {
			flags = append(flags, a)
			if !strings.Contains(a, "=") && takesValue[a] && i+1 < len(args) {
				i++
				flags = append(flags, args[i])
			}
			continue
		}
		positionals = append(positionals, a)
	}
	return append(flags, positionals...)
}

func shouldUseColor(noColorFlag bool, w io.Writer) bool {
	if noColorFlag {
		return false
	}
	if _, set := os.LookupEnv("NO_COLOR"); set {
		return false
	}
	if strings.EqualFold(os.Getenv("TERM"), "dumb") {
		return false
	}
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

func printUsage(w io.Writer) {
	fmt.Fprintf(w, `servecheck v%s — 15-second conformance probe & offline GGUF/Jinja2 doctor for local LLM servers

Usage:
  servecheck <endpoint-url> --model <id> [flags]
  servecheck gguf <path-to-.gguf-or-tokenizer_config.json> [flags]
  servecheck recipes [--engine all|vllm|llama-server|sglang|ollama|mlx] [--model-family <family>] [--format table|json|markdown]
  servecheck version

Server Probe Flags:
  --model <id>            Model ID to test (auto-selected from /models if omitted)
  --api-key <key>         Bearer API key (falls back to $OPENAI_API_KEY)
  --engine <name>         auto|vllm|llama-server|sglang|ollama|mlx (default: auto)
  --timeout <duration>    Per-probe HTTP timeout (default: 20s)
  --format <fmt>          table|json|markdown (default: table)
  --no-color              Disable ANSI colors (also honors NO_COLOR)

Exit Codes:
  0  All probes passed (or warnings only)
  1  One or more probes failed
  2  CLI usage, file, authentication, or connection error

Examples:
  servecheck http://localhost:8000/v1 --model Qwen/Qwen3-32B
  servecheck http://localhost:8080/v1 --model qwen3 --engine llama-server --format markdown
  servecheck gguf ./Qwen3-35B-Q4_K_M.gguf
  servecheck gguf ./tokenizer_config.json --format json
  servecheck recipes --engine vllm
`, version)
}
