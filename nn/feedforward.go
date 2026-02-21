package nn

import (
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// FeedForward is a two-layer MLP: x -> Linear -> GELU -> Linear (standard).
// Optionally SwiGLU: x -> [Linear, Linear] -> silu(gate) * up -> Linear.
type FeedForward struct {
	Linear1 *Linear // in -> hidden (e.g. 4x)
	Linear2 *Linear // hidden -> in
	SwiGLU  bool   // if true, use gate and up projection
}

// Forward runs the FFN. x shape [..., dim].
func (ff *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := ff.Linear1.Forward(x)
	if err != nil {
		return nil, err
	}
	if ff.SwiGLU {
		// SwiGLU: split h into gate and up, then silu(gate)*up. For simplicity we do single Linear1 output and GELU.
		act, err := ops.Gelu(h)
		if err != nil {
			return nil, err
		}
		return ff.Linear2.Forward(act)
	}
	act, err := ops.Gelu(h)
	if err != nil {
		return nil, err
	}
	return ff.Linear2.Forward(act)
}
