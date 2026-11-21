package recipes

import (
	_ "embed"
	"fmt"
	"slices"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

//go:embed recipes.yaml
var embeddedRecipesYAML []byte

// Recipe represents a single engine/model-family remediation entry.
type Recipe struct {
	ID          string   `yaml:"id" json:"id"`
	Engine      string   `yaml:"engine" json:"engine"`
	ModelFamily string   `yaml:"model_family" json:"model_family"`
	Probes      []string `yaml:"probes" json:"probes"`
	Flags       string   `yaml:"flags" json:"flags"`
	Note        string   `yaml:"note" json:"note"`
	MinVersion  string   `yaml:"min_version,omitempty" json:"min_version,omitempty"`
}

// Fix is a matched remediation recommendation ready for reporting.
type Fix struct {
	RecipeID      string   `json:"recipe_id"`
	Engine        string   `json:"engine"`
	ModelFamily   string   `json:"model_family"`
	Flags         string   `json:"flags"`
	Note          string   `json:"note"`
	MinVersion    string   `json:"min_version,omitempty"`
	MatchedProbes []string `json:"matched_probes"`
}

type recipeFile struct {
	Version int      `yaml:"version"`
	Recipes []Recipe `yaml:"recipes"`
}

var (
	loadOnce sync.Once
	cachedDB []Recipe
	loadErr  error
)

// Load parses and returns all embedded remediation recipes.
func Load() ([]Recipe, error) {
	loadOnce.Do(func() {
		var rf recipeFile
		if err := yaml.Unmarshal(embeddedRecipesYAML, &rf); err != nil {
			loadErr = fmt.Errorf("parse embedded recipes.yaml: %w", err)
			return
		}
		cachedDB = rf.Recipes
	})
	if loadErr != nil {
		return nil, loadErr
	}
	out := make([]Recipe, len(cachedDB))
	copy(out, cachedDB)
	return out, nil
}

// InferModelFamily normalizes a model ID or GGUF architecture string into a recipe model family.
func InferModelFamily(modelID, arch string) string {
	combined := strings.ToLower(strings.TrimSpace(modelID + " " + arch))
	switch {
	case strings.Contains(combined, "deepseek"):
		return "deepseek"
	case strings.Contains(combined, "qwen3") || strings.Contains(combined, "qwen-3") || strings.Contains(combined, "qwen_3") || strings.Contains(combined, "qwq"):
		return "qwen3"
	case strings.Contains(combined, "qwen"):
		return "qwen3"
	case strings.Contains(combined, "llama"):
		return "llama"
	case strings.Contains(combined, "gemma"):
		return "gemma"
	case strings.Contains(combined, "mistral") || strings.Contains(combined, "mixtral") || strings.Contains(combined, "devstral") || strings.Contains(combined, "ministral"):
		return "mistral"
	default:
		return "*"
	}
}

// Filter returns recipes matching optional engine and modelFamily filters (empty or "all" matches everything).
func Filter(engine, modelFamily string) ([]Recipe, error) {
	all, err := Load()
	if err != nil {
		return nil, err
	}
	eng := normalizeEngine(engine)
	fam := strings.ToLower(strings.TrimSpace(modelFamily))
	var out []Recipe
	for _, r := range all {
		if eng != "" && eng != "all" && r.Engine != eng {
			continue
		}
		if fam != "" && fam != "all" && r.ModelFamily != fam && r.ModelFamily != "*" {
			continue
		}
		out = append(out, r)
	}
	return out, nil
}

// Match returns deduplicated CLI flag fixes for the given engine, model family, and failing/warning probe IDs.
// When engine is unknown or empty, it returns fixes for both vllm and llama-server (the two primary local servers).
func Match(engine, modelFamily string, probeIDs []string) ([]Fix, error) {
	if len(probeIDs) == 0 {
		return nil, nil
	}
	all, err := Load()
	if err != nil {
		return nil, err
	}

	eng := normalizeEngine(engine)
	engines := []string{eng}
	if eng == "" || eng == "unknown" || eng == "auto" {
		engines = []string{"vllm", "llama-server"}
	}

	fam := strings.ToLower(strings.TrimSpace(modelFamily))
	if fam == "" {
		fam = "*"
	}

	// Deduplicate probeIDs while preserving order.
	uniqProbes := make([]string, 0, len(probeIDs))
	for _, pid := range probeIDs {
		if pid != "" && !slices.Contains(uniqProbes, pid) {
			uniqProbes = append(uniqProbes, pid)
		}
	}

	// Track matched recipes by ID so we merge matched probes cleanly and keep deterministic order.
	fixByID := make(map[string]*Fix)
	order := make([]string, 0)

	for _, targetEngine := range engines {
		for _, pid := range uniqProbes {
			// Prefer exact model_family match; fall back to wildcard "*" for this (engine, probe).
			var exactMatches []Recipe
			var wildcardMatches []Recipe
			for _, r := range all {
				if r.Engine != targetEngine {
					continue
				}
				if !slices.Contains(r.Probes, pid) {
					continue
				}
				if fam != "*" && r.ModelFamily == fam {
					exactMatches = append(exactMatches, r)
				} else if r.ModelFamily == "*" {
					wildcardMatches = append(wildcardMatches, r)
				}
			}

			chosen := exactMatches
			if len(chosen) == 0 {
				chosen = wildcardMatches
			}
			for _, r := range chosen {
				if existing, ok := fixByID[r.ID]; ok {
					if !slices.Contains(existing.MatchedProbes, pid) {
						existing.MatchedProbes = append(existing.MatchedProbes, pid)
					}
					continue
				}
				fixByID[r.ID] = &Fix{
					RecipeID:      r.ID,
					Engine:        r.Engine,
					ModelFamily:   r.ModelFamily,
					Flags:         r.Flags,
					Note:          r.Note,
					MinVersion:    r.MinVersion,
					MatchedProbes: []string{pid},
				}
				order = append(order, r.ID)
			}
		}
	}

	out := make([]Fix, 0, len(order))
	for _, id := range order {
		out = append(out, *fixByID[id])
	}
	return out, nil
}

func normalizeEngine(engine string) string {
	switch strings.ToLower(strings.TrimSpace(engine)) {
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
	case "all":
		return "all"
	case "auto", "unknown", "":
		return strings.ToLower(strings.TrimSpace(engine))
	default:
		return strings.ToLower(strings.TrimSpace(engine))
	}
}
