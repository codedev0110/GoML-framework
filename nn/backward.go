package nn

import (
	"github.com/djeday123/goml/autograd"
	"github.com/djeday123/goml/tensor"
)

// Backward propagates gradients from the loss tensor by invoking the autograd
// helper which executes the GradFn stored on the tensor.
func Backward(loss *tensor.Tensor) {
	if loss == nil {
		return
	}
	autograd.Backward(loss)
}
