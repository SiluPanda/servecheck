package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ggufKVItem is a test helper for constructing synthetic binary GGUF v2/v3 fixtures.
type ggufKVItem struct {
	key   string
	vtype uint32
	val   any
}

func buildSyntheticGGUF(t *testing.T, version uint32, tensorCount uint64, items []ggufKVItem, trailingJunkBytes int) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	_ = binary.Write(&buf, binary.LittleEndian, version)
	_ = binary.Write(&buf, binary.LittleEndian, tensorCount)
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(items)))

	for _, it := range items {
		writeGGUFStr(&buf, it.key)
		_ = binary.Write(&buf, binary.LittleEndian, it.vtype)
		writeGGUFVal(t, &buf, it.vtype, it.val)
	}

	if trailingJunkBytes > 0 {
		buf.Write(make([]byte, trailingJunkBytes))
	}
	return buf.Bytes()
}

func writeGGUFStr(buf *bytes.Buffer, s string) {
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

type ggufArrayVal struct {
	elemType uint32
	elems    []any
}

func writeGGUFVal(t *testing.T, buf *bytes.Buffer, vtype uint32, val any) {
	t.Helper()
	switch vtype {
	case TypeUint8:
		_ = binary.Write(buf, binary.LittleEndian, val.(uint8))
	case TypeInt8:
		_ = binary.Write(buf, binary.LittleEndian, val.(int8))
	case TypeUint16:
		_ = binary.Write(buf, binary.LittleEndian, val.(uint16))
	case TypeInt16:
		_ = binary.Write(buf, binary.LittleEndian, val.(int16))
	case TypeUint32:
		_ = binary.Write(buf, binary.LittleEndian, val.(uint32))
	case TypeInt32:
		_ = binary.Write(buf, binary.LittleEndian, val.(int32))
	case TypeFloat32:
		_ = binary.Write(buf, binary.LittleEndian, math.Float32bits(val.(float32)))
	case TypeBool:
		var b uint8
		if val.(bool) {
			b = 1
		}
		_ = binary.Write(buf, binary.LittleEndian, b)
	case TypeString:
		writeGGUFStr(buf, val.(string))
	case TypeUint64:
		_ = binary.Write(buf, binary.LittleEndian, val.(uint64))
	case TypeInt64:
		_ = binary.Write(buf, binary.LittleEndian, val.(int64))
	case TypeFloat64:
		_ = binary.Write(buf, binary.LittleEndian, math.Float64bits(val.(float64)))
	case TypeArray:
		av := val.(ggufArrayVal)
		_ = binary.Write(buf, binary.LittleEndian, av.elemType)
		_ = binary.Write(buf, binary.LittleEndian, uint64(len(av.elems)))
		for _, e := range av.elems {
			writeGGUFVal(t, buf, av.elemType, e)
		}
	default:
		t.Fatalf("unsupported test vtype %d", vtype)
	}
}

func TestParseGGUFv3AllValueTypesAndNestedArrays(t *testing.T) {
	const sampleTmpl = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

	merges := make([]any, 5000)
	for i := range merges {
		merges[i] = "a b"
	}

	items := []ggufKVItem{
		{"general.architecture", TypeString, "qwen3"},
		{"general.name", TypeString, "Qwen3-35B-A3B"},
		{"test.u8", TypeUint8, uint8(200)},
		{"test.i8", TypeInt8, int8(-42)},
		{"test.u16", TypeUint16, uint16(65000)},
		{"test.i16", TypeInt16, int16(-1234)},
		{"test.u32", TypeUint32, uint32(123456)},
		{"test.i32", TypeInt32, int32(-98765)},
		{"test.f32", TypeFloat32, float32(3.5)},
		{"test.bool", TypeBool, true},
		{"test.u64", TypeUint64, uint64(9999999999)},
		{"test.i64", TypeInt64, int64(-8888888888)},
		{"test.f64", TypeFloat64, float64(2.718281828)},
		{"test.nested_array", TypeArray, ggufArrayVal{
			elemType: TypeArray,
			elems: []any{
				ggufArrayVal{elemType: TypeUint32, elems: []any{uint32(1), uint32(2)}},
				ggufArrayVal{elemType: TypeUint32, elems: []any{uint32(3), uint32(4)}},
			},
		}},
		{"tokenizer.ggml.tokens", TypeArray, ggufArrayVal{
			elemType: TypeString,
			elems:    []any{"<unk>", "<|im_start|>", "<|im_end|>"},
		}},
		{"tokenizer.ggml.merges", TypeArray, ggufArrayVal{
			elemType: TypeString,
			elems:    merges,
		}},
		{"tokenizer.ggml.bos_token_id", TypeUint32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", TypeUint32, uint32(2)},
		{"tokenizer.chat_template", TypeString, sampleTmpl},
		{"tokenizer.chat_template.tool_use", TypeString, "{{ tools | tojson }}"},
	}

	// Append 4 MiB of fake tensor weights to prove parser stops immediately after KV metadata.
	raw := buildSyntheticGGUF(t, 3, 512, items, 4<<20)

	tmpPath := filepath.Join(t.TempDir(), "qwen3-test.gguf")
	if err := os.WriteFile(tmpPath, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	meta, err := LoadFile(tmpPath)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("LoadFile() error: %v", err)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("LoadFile took too long: %v", elapsed)
	}

	if meta.Format != "gguf-v3" || meta.Version != 3 || meta.TensorCount != 512 {
		t.Fatalf("unexpected header: %+v", meta)
	}
	if meta.Architecture != "qwen3" || meta.Name != "Qwen3-35B-A3B" {
		t.Fatalf("unexpected arch/name: %q / %q", meta.Architecture, meta.Name)
	}
	if meta.BOSToken != "<|im_start|>" || meta.EOSToken != "<|im_end|>" {
		t.Fatalf("unexpected BOS/EOS: %q / %q", meta.BOSToken, meta.EOSToken)
	}
	if meta.ChatTemplate != sampleTmpl {
		t.Fatalf("chat_template mismatch: got %q", meta.ChatTemplate)
	}
	if meta.NamedTemplates["tool_use"] != "{{ tools | tojson }}" {
		t.Fatalf("named template mismatch: %+v", meta.NamedTemplates)
	}
	if meta.KV["test.u8"] != uint8(200) || meta.KV["test.i64"] != int64(-8888888888) || meta.KV["test.bool"] != true {
		t.Fatalf("scalar KV mismatch: %+v", meta.KV)
	}
	nested, ok := meta.KV["test.nested_array"].([]any)
	if !ok || len(nested) != 2 {
		t.Fatalf("nested array mismatch: %#v", meta.KV["test.nested_array"])
	}
	row1, ok := nested[1].([]any)
	if !ok || len(row1) != 2 || row1[1] != uint32(4) {
		t.Fatalf("nested array element mismatch: %#v", nested)
	}
}

func TestParseGGUFv2AndErrors(t *testing.T) {
	rawV2 := buildSyntheticGGUF(t, 2, 16, []ggufKVItem{
		{"general.architecture", TypeString, "llama"},
		{"tokenizer.chat_template", TypeString, "{{ bos_token }}hi"},
	}, 0)
	meta, err := ParseGGUF(bytes.NewReader(rawV2))
	if err != nil {
		t.Fatalf("ParseGGUF v2 error: %v", err)
	}
	if meta.Format != "gguf-v2" || meta.Architecture != "llama" {
		t.Fatalf("unexpected v2 meta: %+v", meta)
	}

	// Unsupported v1
	rawV1 := buildSyntheticGGUF(t, 1, 0, nil, 0)
	if _, err := ParseGGUF(bytes.NewReader(rawV1)); err == nil {
		t.Fatal("expected error for unsupported GGUF v1")
	}

	// Bad magic
	if _, err := ParseGGUF(bytes.NewReader([]byte("NOPE\x03\x00\x00\x00"))); err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestLoadTokenizerConfigJSON(t *testing.T) {
	dir := t.TempDir()

	t.Run("string_template_and_added_token_objects", func(t *testing.T) {
		p := filepath.Join(dir, "tokenizer_config.json")
		data := `{
			"model_type": "qwen3",
			"name_or_path": "Qwen/Qwen3-32B",
			"bos_token": {"content": "<|im_start|>"},
			"eos_token": {"content": "<|im_end|>"},
			"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
		}`
		if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
		meta, err := LoadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if meta.Format != "tokenizer_config.json" || meta.BOSToken != "<|im_start|>" || meta.EOSToken != "<|im_end|>" {
			t.Fatalf("unexpected meta: %+v", meta)
		}
		if meta.ChatTemplate == "" {
			t.Fatal("expected non-empty chat_template")
		}
	})

	t.Run("named_template_list", func(t *testing.T) {
		p := filepath.Join(dir, "named_tokenizer_config.json")
		data := `{
			"tokenizer_class": "LlamaTokenizerFast",
			"bos_token": "<|begin_of_text|>",
			"eos_token": "<|eot_id|>",
			"chat_template": [
				{"name": "default", "template": "{{ bos_token }}default"},
				{"name": "tool_use", "template": "{{ bos_token }}tools"}
			]
		}`
		if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
		meta, err := LoadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if meta.ChatTemplate != "{{ bos_token }}default" || meta.NamedTemplates["tool_use"] != "{{ bos_token }}tools" {
			t.Fatalf("unexpected named templates: %+v", meta)
		}
	})
}
