package nn

import "github.com/djeday123/goml/tensor"

// Backward propagates gradients through the model. For the full LLM we would
// traverse layers in reverse and call each layer's backward (Linear backward:
// grad_W = x^T @ grad_out, grad_x = grad_out @ W; etc.).
// This file is a placeholder for manual backward; the actual backward for
// training can be implemented layer-by-layer when needed.
// Many operations already set tensor.Backward in ops/ops.go.
func Backward(loss *tensor.Tensor) {}
