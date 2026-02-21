package tokenizer

// ByteLevel is the simplest tokenizer: each byte is a token. Vocab size 256.
type ByteLevel struct {
	VocabSize int
}

// NewByteLevel returns a byte-level tokenizer.
func NewByteLevel() *ByteLevel {
	return &ByteLevel{VocabSize: 256}
}

// Encode converts a string to token ids (one byte = one id).
func (b *ByteLevel) Encode(s string) []int64 {
	out := make([]int64, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = int64(s[i])
	}
	return out
}

// Decode converts token ids back to string.
func (b *ByteLevel) Decode(ids []int64) string {
	out := make([]byte, len(ids))
	for i, id := range ids {
		if id >= 0 && id < 256 {
			out[i] = byte(id)
		}
	}
	return string(out)
}
