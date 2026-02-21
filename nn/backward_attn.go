package nn

import "github.com/djeday123/goml/tensor"

// ForwardWithCache runs attention forward and stores (q, k, v, attn_weights) for backward.
// Backward recomputes or uses cached values to compute gradients for Q, K, V projections.
// This file is a placeholder; full implementation would cache intermediate tensors
// and implement dL/dQ, dL/dK, dL/dV from dL/dOut.
func ForwardWithCache(a *Attention, x *tensor.Tensor) (*tensor.Tensor, interface{}) {
	out, _ := a.Forward(x)
	return out, nil
}
