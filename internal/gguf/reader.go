package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// GGUF metadata value type constants per the GGML GGUF v2/v3 specification.
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

const (
	maxGGUFStringBytes = 32 << 20 // 32 MiB safety cap for any single metadata string
	maxGGUFKVCount     = 1 << 20  // 1M KV entries safety cap
	maxGGUFArrayLen    = 4 << 20  // 4M array elements safety cap
	maxGGUFArrayDepth  = 16       // nested array recursion depth cap
	retainArrayLimit   = 4096     // retain arbitrary arrays up to this length in KV map
	retainVocabLimit   = 300000   // retain tokenizer.ggml.tokens up to 300k for BOS/EOS resolution
)

// Metadata contains parsed header and tokenizer metadata from a GGUF file or tokenizer_config.json.
type Metadata struct {
	Path            string            `json:"path,omitempty"`
	Format          string            `json:"format"`
	Version         uint32            `json:"version,omitempty"`
	TensorCount     uint64            `json:"tensor_count,omitempty"`
	MetadataKVCount uint64            `json:"metadata_kv_count,omitempty"`
	Architecture    string            `json:"architecture,omitempty"`
	Name            string            `json:"name,omitempty"`
	ChatTemplate    string            `json:"chat_template,omitempty"`
	NamedTemplates  map[string]string `json:"named_templates,omitempty"`
	BOSToken        string            `json:"bos_token,omitempty"`
	EOSToken        string            `json:"eos_token,omitempty"`
	BOSTokenID      *uint64           `json:"bos_token_id,omitempty"`
	EOSTokenID      *uint64           `json:"eos_token_id,omitempty"`
	KV              map[string]any    `json:"-"`
}

// LoadFile opens a .gguf binary or HuggingFace tokenizer_config.json file and extracts its metadata.
// For GGUF files it reads only the header and metadata KV section without reading tensor weights.
func LoadFile(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 64*1024)
	peek, err := br.Peek(4)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("read header of %s: %w", path, err)
	}

	if len(peek) == 4 && string(peek) == "GGUF" {
		meta, err := ParseGGUF(br)
		if err != nil {
			return nil, fmt.Errorf("parse GGUF %s: %w", path, err)
		}
		meta.Path = path
		if meta.Name == "" {
			meta.Name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		}
		return meta, nil
	}

	// Detect JSON (tokenizer_config.json) by extension or leading '{'.
	trimmedPeek := bytes.TrimSpace(peek)
	if strings.HasSuffix(strings.ToLower(path), ".json") || (len(trimmedPeek) > 0 && trimmedPeek[0] == '{') {
		raw, err := io.ReadAll(io.LimitReader(br, maxGGUFStringBytes))
		if err != nil {
			return nil, fmt.Errorf("read JSON %s: %w", path, err)
		}
		meta, err := ParseTokenizerConfigJSON(raw)
		if err != nil {
			return nil, fmt.Errorf("parse tokenizer_config.json %s: %w", path, err)
		}
		meta.Path = path
		if meta.Name == "" {
			dirName := filepath.Base(filepath.Dir(path))
			if dirName != "." && dirName != "/" && dirName != "" {
				meta.Name = dirName
			} else {
				meta.Name = filepath.Base(path)
			}
		}
		return meta, nil
	}

	return nil, fmt.Errorf("%s: unrecognized file format (expected GGUF magic or JSON tokenizer_config)", path)
}

// ParseGGUF parses a GGUF v2 or v3 binary header and metadata KV section from r.
// It stops immediately after the metadata section and never reads tensor data.
func ParseGGUF(r io.Reader) (*Metadata, error) {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReaderSize(r, 64*1024)
	}

	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("invalid GGUF magic %q (expected \"GGUF\")", string(magic[:]))
	}

	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d (supported: v2, v3)", version)
	}

	var tensorCount, kvCount uint64
	if err := binary.Read(br, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor_count: %w", err)
	}
	if err := binary.Read(br, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("read metadata_kv_count: %w", err)
	}
	if kvCount > maxGGUFKVCount {
		return nil, fmt.Errorf("metadata_kv_count %d exceeds safety cap %d", kvCount, maxGGUFKVCount)
	}

	meta := &Metadata{
		Format:          fmt.Sprintf("gguf-v%d", version),
		Version:         version,
		TensorCount:     tensorCount,
		MetadataKVCount: kvCount,
		NamedTemplates:  map[string]string{},
		KV:              make(map[string]any, min(int(kvCount), 256)),
	}

	var vocabTokens []string

	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(br)
		if err != nil {
			return nil, fmt.Errorf("kv[%d] key: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(br, binary.LittleEndian, &vtype); err != nil {
			return nil, fmt.Errorf("kv[%d] (%s) value_type: %w", i, key, err)
		}

		// Fast-path skip huge non-essential tokenizer tables (merges, scores, token_type).
		if vtype == TypeArray && shouldSkipLargeArrayKey(key) {
			if err := skipGGUFValue(br, vtype, 0); err != nil {
				return nil, fmt.Errorf("kv[%d] (%s) skip array: %w", i, key, err)
			}
			continue
		}

		// Special-case tokenizer.ggml.tokens so we can resolve BOS/EOS token strings quickly.
		if vtype == TypeArray && key == "tokenizer.ggml.tokens" {
			toks, err := readStringArrayLimited(br, retainVocabLimit)
			if err != nil {
				return nil, fmt.Errorf("kv[%d] (%s): %w", i, key, err)
			}
			vocabTokens = toks
			continue
		}

		val, retained, err := readGGUFValue(br, vtype, 0)
		if err != nil {
			return nil, fmt.Errorf("kv[%d] (%s) type=%d: %w", i, key, vtype, err)
		}
		if retained {
			meta.KV[key] = val
		}

		switch key {
		case "general.architecture":
			if s, ok := val.(string); ok {
				meta.Architecture = s
			}
		case "general.name":
			if s, ok := val.(string); ok {
				meta.Name = s
			}
		case "tokenizer.chat_template":
			if s, ok := val.(string); ok {
				meta.ChatTemplate = s
			}
		case "tokenizer.ggml.bos_token_id":
			if id, ok := toUint64(val); ok {
				meta.BOSTokenID = &id
			}
		case "tokenizer.ggml.eos_token_id":
			if id, ok := toUint64(val); ok {
				meta.EOSTokenID = &id
			}
		default:
			if strings.HasPrefix(key, "tokenizer.chat_template.") {
				if s, ok := val.(string); ok {
					name := strings.TrimPrefix(key, "tokenizer.chat_template.")
					meta.NamedTemplates[name] = s
				}
			}
		}
	}

	// Resolve BOS/EOS strings from vocab table when IDs are present.
	if meta.BOSTokenID != nil && *meta.BOSTokenID < uint64(len(vocabTokens)) {
		meta.BOSToken = vocabTokens[*meta.BOSTokenID]
	}
	if meta.EOSTokenID != nil && *meta.EOSTokenID < uint64(len(vocabTokens)) {
		meta.EOSToken = vocabTokens[*meta.EOSTokenID]
	}

	if len(meta.NamedTemplates) == 0 {
		meta.NamedTemplates = nil
	}
	return meta, nil
}

// ParseTokenizerConfigJSON extracts chat_template and BOS/EOS metadata from a HuggingFace tokenizer_config.json payload.
func ParseTokenizerConfigJSON(data []byte) (*Metadata, error) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}

	meta := &Metadata{
		Format:         "tokenizer_config.json",
		NamedTemplates: map[string]string{},
		KV:             raw,
	}

	if s, ok := raw["model_type"].(string); ok {
		meta.Architecture = s
	} else if s, ok := raw["tokenizer_class"].(string); ok {
		meta.Architecture = s
	}
	if s, ok := raw["name_or_path"].(string); ok && s != "" {
		meta.Name = s
	}

	meta.BOSToken = extractHFTokenString(raw["bos_token"])
	meta.EOSToken = extractHFTokenString(raw["eos_token"])

	switch ct := raw["chat_template"].(type) {
	case string:
		meta.ChatTemplate = ct
	case []any:
		// HuggingFace named template list: [{"name": "default", "template": "..."}, {"name": "tool_use", "template": "..."}]
		for _, item := range ct {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}
			name, _ := obj["name"].(string)
			tmpl, _ := obj["template"].(string)
			if tmpl == "" {
				continue
			}
			if name != "" {
				meta.NamedTemplates[name] = tmpl
			}
			if meta.ChatTemplate == "" || name == "default" || name == "tool_use" {
				if name == "default" || meta.ChatTemplate == "" {
					meta.ChatTemplate = tmpl
				}
			}
		}
	}

	if len(meta.NamedTemplates) == 0 {
		meta.NamedTemplates = nil
	}
	return meta, nil
}

func extractHFTokenString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case map[string]any:
		if content, ok := t["content"].(string); ok {
			return content
		}
	}
	return ""
}

func shouldSkipLargeArrayKey(key string) bool {
	switch key {
	case "tokenizer.ggml.merges", "tokenizer.ggml.scores", "tokenizer.ggml.token_type":
		return true
	default:
		return false
	}
}

func readGGUFString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > maxGGUFStringBytes {
		return "", fmt.Errorf("string length %d exceeds cap %d", n, maxGGUFStringBytes)
	}
	if n == 0 {
		return "", nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func skipGGUFString(r io.Reader) error {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return err
	}
	if n > maxGGUFStringBytes {
		return fmt.Errorf("string length %d exceeds cap %d", n, maxGGUFStringBytes)
	}
	if n == 0 {
		return nil
	}
	_, err := io.CopyN(io.Discard, r, int64(n))
	return err
}

func readStringArrayLimited(r io.Reader, limit uint64) ([]string, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, err
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if count > maxGGUFArrayLen {
		return nil, fmt.Errorf("array length %d exceeds cap %d", count, maxGGUFArrayLen)
	}
	if elemType != TypeString {
		for i := uint64(0); i < count; i++ {
			if err := skipGGUFValue(r, elemType, 1); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}

	retain := count <= limit
	var out []string
	if retain {
		out = make([]string, 0, count)
	}
	for i := uint64(0); i < count; i++ {
		if retain {
			s, err := readGGUFString(r)
			if err != nil {
				return nil, err
			}
			out = append(out, s)
		} else {
			if err := skipGGUFString(r); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func readGGUFValue(r io.Reader, vtype uint32, depth int) (any, bool, error) {
	switch vtype {
	case TypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeFloat32:
		var bits uint32
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, false, err
		}
		return math.Float32frombits(bits), true, nil
	case TypeBool:
		var b uint8
		if err := binary.Read(r, binary.LittleEndian, &b); err != nil {
			return nil, false, err
		}
		return b != 0, true, nil
	case TypeString:
		s, err := readGGUFString(r)
		return s, true, err
	case TypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, true, err
	case TypeFloat64:
		var bits uint64
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, false, err
		}
		return math.Float64frombits(bits), true, nil
	case TypeArray:
		if depth >= maxGGUFArrayDepth {
			return nil, false, fmt.Errorf("array nesting depth exceeds %d", maxGGUFArrayDepth)
		}
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, false, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, false, err
		}
		if count > maxGGUFArrayLen {
			return nil, false, fmt.Errorf("array length %d exceeds cap %d", count, maxGGUFArrayLen)
		}
		if count > retainArrayLimit {
			for i := uint64(0); i < count; i++ {
				if err := skipGGUFValue(r, elemType, depth+1); err != nil {
					return nil, false, err
				}
			}
			return nil, false, nil
		}
		arr := make([]any, 0, count)
		for i := uint64(0); i < count; i++ {
			elem, _, err := readGGUFValue(r, elemType, depth+1)
			if err != nil {
				return nil, false, err
			}
			arr = append(arr, elem)
		}
		return arr, true, nil
	default:
		return nil, false, fmt.Errorf("unknown GGUF metadata value type %d", vtype)
	}
}

func skipGGUFValue(r io.Reader, vtype uint32, depth int) error {
	if size := fixedScalarSize(vtype); size > 0 {
		_, err := io.CopyN(io.Discard, r, size)
		return err
	}
	switch vtype {
	case TypeString:
		return skipGGUFString(r)
	case TypeArray:
		if depth >= maxGGUFArrayDepth {
			return fmt.Errorf("array nesting depth exceeds %d", maxGGUFArrayDepth)
		}
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
		if count > maxGGUFArrayLen {
			return fmt.Errorf("array length %d exceeds cap %d", count, maxGGUFArrayLen)
		}
		// Fast O(1) skip for fixed-width scalar arrays!
		if elemSize := fixedScalarSize(elemType); elemSize > 0 {
			total := count * uint64(elemSize)
			if total > (1 << 34) {
				return fmt.Errorf("array byte length %d overflow", total)
			}
			_, err := io.CopyN(io.Discard, r, int64(total))
			return err
		}
		for i := uint64(0); i < count; i++ {
			if err := skipGGUFValue(r, elemType, depth+1); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown GGUF metadata value type %d", vtype)
	}
}

func fixedScalarSize(vtype uint32) int64 {
	switch vtype {
	case TypeUint8, TypeInt8, TypeBool:
		return 1
	case TypeUint16, TypeInt16:
		return 2
	case TypeUint32, TypeInt32, TypeFloat32:
		return 4
	case TypeUint64, TypeInt64, TypeFloat64:
		return 8
	default:
		return 0
	}
}

func toUint64(v any) (uint64, bool) {
	switch n := v.(type) {
	case uint8:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint32:
		return uint64(n), true
	case uint64:
		return n, true
	case int8:
		if n >= 0 {
			return uint64(n), true
		}
	case int16:
		if n >= 0 {
			return uint64(n), true
		}
	case int32:
		if n >= 0 {
			return uint64(n), true
		}
	case int64:
		if n >= 0 {
			return uint64(n), true
		}
	}
	return 0, false
}
