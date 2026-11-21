package report

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/servecheck/servecheck/internal/gguf"
	"github.com/servecheck/servecheck/internal/probe"
	"github.com/servecheck/servecheck/internal/recipes"
)

// Format represents a supported output rendering format.
type Format string

const (
	FormatTable    Format = "table"
	FormatJSON     Format = "json"
	FormatMarkdown Format = "markdown"
)

// ParseFormat validates and normalizes a --format flag value.
func ParseFormat(s string) (Format, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "table", "text":
		return FormatTable, nil
	case "json":
		return FormatJSON, nil
	case "markdown", "md":
		return FormatMarkdown, nil
	default:
		return "", fmt.Errorf("invalid --format %q (expected table, json, or markdown)", s)
	}
}

// Options controls report rendering.
type Options struct {
	Format  Format
	Color   bool
	Version string
}

const (
	ansiReset  = "\033[0m"
	ansiBold   = "\033[1m"
	ansiDim    = "\033[2m"
	ansiRed    = "\033[31m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
	ansiCyan   = "\033[36m"
)

func (o Options) style(code, text string) string {
	if !o.Color {
		return text
	}
	return code + text + ansiReset
}

func (o Options) badge(st probe.Status) string {
	switch st {
	case probe.StatusPass:
		return o.style(ansiGreen+ansiBold, "[PASS]")
	case probe.StatusWarn:
		return o.style(ansiYellow+ansiBold, "[WARN]")
	case probe.StatusFail:
		return o.style(ansiRed+ansiBold, "[FAIL]")
	default:
		return "[" + string(st) + "]"
	}
}

type serverJSONPayload struct {
	Version string `json:"version"`
	*probe.RunResult
	Fixes []recipes.Fix `json:"recommended_fixes,omitempty"`
}

type ggufJSONPayload struct {
	Version string `json:"version"`
	*gguf.DoctorReport
	Fixes []recipes.Fix `json:"recommended_fixes,omitempty"`
}

// FormatServerReport writes a server conformance probe report in table, markdown, or json format.
func FormatServerReport(w io.Writer, res *probe.RunResult, fixes []recipes.Fix, opts Options) error {
	switch opts.Format {
	case FormatJSON:
		payload := serverJSONPayload{
			Version:   opts.Version,
			RunResult: res,
			Fixes:     fixes,
		}
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		return enc.Encode(payload)
	case FormatMarkdown:
		return writeServerMarkdown(w, res, fixes, opts)
	default:
		return writeServerTable(w, res, fixes, opts)
	}
}

// FormatGGUFReport writes an offline GGUF/Jinja2 doctor report in table, markdown, or json format.
func FormatGGUFReport(w io.Writer, rep *gguf.DoctorReport, fixes []recipes.Fix, opts Options) error {
	switch opts.Format {
	case FormatJSON:
		payload := ggufJSONPayload{
			Version:      opts.Version,
			DoctorReport: rep,
			Fixes:        fixes,
		}
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		return enc.Encode(payload)
	case FormatMarkdown:
		return writeGGUFMarkdown(w, rep, fixes, opts)
	default:
		return writeGGUFTable(w, rep, fixes, opts)
	}
}

// FormatRecipes writes the embedded remediation recipes catalog in table, markdown, or json format.
func FormatRecipes(w io.Writer, list []recipes.Recipe, opts Options) error {
	switch opts.Format {
	case FormatJSON:
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		return enc.Encode(list)
	case FormatMarkdown:
		fmt.Fprintln(w, "# servecheck Remediation Recipes")
		fmt.Fprintln(w)
		fmt.Fprintln(w, "| Engine | Family | Probes | CLI Flags | Notes |")
		fmt.Fprintln(w, "|---|---|---|---|---|")
		for _, r := range list {
			fmt.Fprintf(w, "| `%s` | `%s` | `%s` | `%s` | %s |\n",
				r.Engine, r.ModelFamily, strings.Join(r.Probes, ", "), escapeMD(r.Flags), escapeMD(r.Note))
		}
		return nil
	default:
		if len(list) == 0 {
			fmt.Fprintln(w, "No matching recipes found.")
			return nil
		}
		fmt.Fprintln(w, opts.style(ansiBold, fmt.Sprintf("servecheck recipes (%d entries)", len(list))))
		fmt.Fprintln(w)
		curEngine := ""
		for _, r := range list {
			if r.Engine != curEngine {
				curEngine = r.Engine
				fmt.Fprintln(w, opts.style(ansiCyan+ansiBold, fmt.Sprintf("== %s ==", probe.DisplayEngineName(curEngine))))
			}
			fam := r.ModelFamily
			if fam == "*" {
				fam = "generic"
			}
			fmt.Fprintf(w, "  %s %s\n", opts.style(ansiBold, r.ID), opts.style(ansiDim, fmt.Sprintf("(family=%s, probes=%s)", fam, strings.Join(r.Probes, ","))))
			fmt.Fprintf(w, "    + %s\n", opts.style(ansiGreen, r.Flags))
			if r.Note != "" {
				fmt.Fprintf(w, "      %s\n", opts.style(ansiDim, r.Note))
			}
		}
		return nil
	}
}

func writeServerTable(w io.Writer, res *probe.RunResult, fixes []recipes.Fix, opts Options) error {
	ver := opts.Version
	if ver == "" {
		ver = "0.1.0"
	}
	fmt.Fprintf(w, "%s %s (%s)\n",
		opts.style(ansiBold, "servecheck v"+ver),
		opts.style(ansiCyan, res.Endpoint),
		res.Model,
	)

	det := res.Engine.Display()
	if res.Engine.DetectedFrom != "" {
		det += " [" + res.Engine.DetectedFrom + "]"
	}
	fmt.Fprintf(w, "Detected engine: %s | Elapsed: %.2fs\n\n",
		opts.style(ansiBold, det),
		float64(res.ElapsedMs)/1000.0,
	)

	writeProbeTableRows(w, res.Results, opts)
	writeFixesTable(w, fixes, res.Engine.Engine, res.ModelFamily, opts)

	fmt.Fprintf(w, "\nSummary: %s, %s, %s (exit %d)\n",
		opts.style(ansiGreen+ansiBold, fmt.Sprintf("%d PASS", res.Counts.Pass)),
		opts.style(ansiYellow+ansiBold, fmt.Sprintf("%d WARN", res.Counts.Warn)),
		opts.style(ansiRed+ansiBold, fmt.Sprintf("%d FAIL", res.Counts.Fail)),
		res.ExitCode(),
	)
	return nil
}

func writeGGUFTable(w io.Writer, rep *gguf.DoctorReport, fixes []recipes.Fix, opts Options) error {
	ver := opts.Version
	if ver == "" {
		ver = "0.1.0"
	}
	target := rep.Path
	if target == "" {
		target = rep.ModelName
	}
	fmt.Fprintf(w, "%s %s (%s)\n",
		opts.style(ansiBold, "servecheck gguf v"+ver),
		opts.style(ansiCyan, target),
		rep.Format,
	)

	metaBits := []string{}
	if rep.ModelName != "" {
		metaBits = append(metaBits, "model="+rep.ModelName)
	}
	if rep.Architecture != "" {
		metaBits = append(metaBits, "arch="+rep.Architecture)
	}
	if rep.TensorCount > 0 {
		metaBits = append(metaBits, fmt.Sprintf("tensors=%d", rep.TensorCount))
	}
	if rep.KVCount > 0 {
		metaBits = append(metaBits, fmt.Sprintf("kv=%d", rep.KVCount))
	}
	if rep.BOSToken != "" {
		metaBits = append(metaBits, fmt.Sprintf("bos=%q", rep.BOSToken))
	}
	if rep.EOSToken != "" {
		metaBits = append(metaBits, fmt.Sprintf("eos=%q", rep.EOSToken))
	}
	metaBits = append(metaBits, fmt.Sprintf("elapsed=%dms", rep.ElapsedMs))
	fmt.Fprintf(w, "Metadata: %s\n\n", opts.style(ansiDim, strings.Join(metaBits, " | ")))

	writeProbeTableRows(w, rep.Results, opts)
	writeFixesTable(w, fixes, "", rep.ModelFamily, opts)

	fmt.Fprintf(w, "\nSummary: %s, %s, %s (exit %d)\n",
		opts.style(ansiGreen+ansiBold, fmt.Sprintf("%d PASS", rep.Counts.Pass)),
		opts.style(ansiYellow+ansiBold, fmt.Sprintf("%d WARN", rep.Counts.Warn)),
		opts.style(ansiRed+ansiBold, fmt.Sprintf("%d FAIL", rep.Counts.Fail)),
		rep.ExitCode(),
	)
	return nil
}

func writeProbeTableRows(w io.Writer, results []probe.ProbeResult, opts Options) {
	maxID := 0
	for _, r := range results {
		if len(r.ID) > maxID {
			maxID = len(r.ID)
		}
	}
	if maxID < 26 {
		maxID = 26
	}

	for _, r := range results {
		pad := strings.Repeat(" ", max(1, maxID-len(r.ID)+2))
		causePrefix := ""
		if r.Cause != "" {
			color := ansiYellow
			if r.Status == probe.StatusFail {
				color = ansiRed
			}
			causePrefix = opts.style(color, "["+string(r.Cause)+"]") + " "
		}
		fmt.Fprintf(w, "  %s %s%s%s%s\n",
			opts.badge(r.Status),
			opts.style(ansiBold, r.ID),
			pad,
			causePrefix,
			r.Summary,
		)
		if r.Details != "" && r.Status != probe.StatusPass {
			indent := strings.Repeat(" ", maxID+11)
			fmt.Fprintf(w, "%s%s\n", indent, opts.style(ansiDim, "↳ "+r.Details))
		}
	}
}

func writeFixesTable(w io.Writer, fixes []recipes.Fix, detectedEngine, modelFamily string, opts Options) {
	if len(fixes) == 0 {
		return
	}
	header := "Recommended server fixes"
	if detectedEngine != "" && detectedEngine != probe.EngineUnknown && detectedEngine != probe.EngineAuto {
		header += fmt.Sprintf(" (%s", probe.DisplayEngineName(detectedEngine))
		if modelFamily != "" && modelFamily != "*" {
			header += " / " + modelFamily
		}
		header += ")"
	} else if modelFamily != "" && modelFamily != "*" {
		header += fmt.Sprintf(" (family: %s)", modelFamily)
	}
	fmt.Fprintf(w, "\n%s:\n", opts.style(ansiBold, header))

	multipleEngines := false
	firstEng := fixes[0].Engine
	for _, f := range fixes[1:] {
		if f.Engine != firstEng {
			multipleEngines = true
			break
		}
	}
	for _, f := range fixes {
		prefix := "  + "
		if multipleEngines {
			prefix = fmt.Sprintf("  [%s] + ", probe.DisplayEngineName(f.Engine))
		}
		fmt.Fprintf(w, "%s%s\n", opts.style(ansiCyan+ansiBold, prefix), opts.style(ansiGreen, f.Flags))
		if f.Note != "" {
			fmt.Fprintf(w, "      %s\n", opts.style(ansiDim, f.Note+" (fixes: "+strings.Join(f.MatchedProbes, ", ")+")"))
		}
	}
}

func writeServerMarkdown(w io.Writer, res *probe.RunResult, fixes []recipes.Fix, opts Options) error {
	fmt.Fprintf(w, "## servecheck Report — `%s`\n\n", res.Model)
	fmt.Fprintf(w, "- **Endpoint:** `%s`\n", res.Endpoint)
	fmt.Fprintf(w, "- **Detected engine:** %s\n", res.Engine.Display())
	fmt.Fprintf(w, "- **Elapsed:** %.2fs\n", float64(res.ElapsedMs)/1000.0)
	fmt.Fprintf(w, "- **Summary:** **%d PASS**, **%d WARN**, **%d FAIL** (exit `%d`)\n\n",
		res.Counts.Pass, res.Counts.Warn, res.Counts.Fail, res.ExitCode())

	writeProbeMarkdownTable(w, res.Results)
	writeFixesMarkdown(w, fixes)
	return nil
}

func writeGGUFMarkdown(w io.Writer, rep *gguf.DoctorReport, fixes []recipes.Fix, opts Options) error {
	target := rep.Path
	if target == "" {
		target = rep.ModelName
	}
	fmt.Fprintf(w, "## servecheck GGUF / Jinja2 Doctor — `%s`\n\n", target)
	fmt.Fprintf(w, "- **Format:** `%s`\n", rep.Format)
	if rep.Architecture != "" {
		fmt.Fprintf(w, "- **Architecture:** `%s`\n", rep.Architecture)
	}
	if rep.ModelName != "" {
		fmt.Fprintf(w, "- **Model:** `%s`\n", rep.ModelName)
	}
	fmt.Fprintf(w, "- **Elapsed:** %dms\n", rep.ElapsedMs)
	fmt.Fprintf(w, "- **Summary:** **%d PASS**, **%d WARN**, **%d FAIL** (exit `%d`)\n\n",
		rep.Counts.Pass, rep.Counts.Warn, rep.Counts.Fail, rep.ExitCode())

	writeProbeMarkdownTable(w, rep.Results)
	writeFixesMarkdown(w, fixes)
	return nil
}

func writeProbeMarkdownTable(w io.Writer, results []probe.ProbeResult) {
	fmt.Fprintln(w, "| Status | Check | Cause | Summary | Time |")
	fmt.Fprintln(w, "|---|---|---|---|---:|")
	for _, r := range results {
		cause := "-"
		if r.Cause != "" {
			cause = "`" + string(r.Cause) + "`"
		}
		summary := r.Summary
		if r.Details != "" && r.Status != probe.StatusPass {
			summary += " — " + r.Details
		}
		fmt.Fprintf(w, "| **%s** | `%s` | %s | %s | %dms |\n",
			r.Status, r.ID, cause, escapeMD(summary), r.DurationMs)
	}
}

func writeFixesMarkdown(w io.Writer, fixes []recipes.Fix) {
	if len(fixes) == 0 {
		return
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "### Recommended Server CLI Fixes")
	fmt.Fprintln(w)
	for _, f := range fixes {
		fmt.Fprintf(w, "**%s** (`%s` — fixes %s):\n",
			probe.DisplayEngineName(f.Engine), f.RecipeID, strings.Join(f.MatchedProbes, ", "))
		fmt.Fprintln(w, "```bash")
		fmt.Fprintln(w, f.Flags)
		fmt.Fprintln(w, "```")
		if f.Note != "" {
			fmt.Fprintf(w, "> %s\n\n", f.Note)
		}
	}
}

func escapeMD(s string) string {
	s = strings.ReplaceAll(s, "|", `\|`)
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
