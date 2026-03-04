package nn

import "github.com/djeday123/goml/tensor"

// Backward propagates gradients from the loss tensor. It calls the Backward
// closure on the loss (which was set by CrossEntropyLoss to fill logits.Grad),
// and then recursively calls Backward on any tensor that has a Backward closure.
func Backward(loss *tensor.Tensor) {
	if loss == nil {
		return
	}
	if loss.Backward != nil {
		loss.Backward()
	}
}
