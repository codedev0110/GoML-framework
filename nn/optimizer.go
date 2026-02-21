package nn

import (
	"github.com/djeday123/goml/optim"
	"github.com/djeday123/goml/tensor"
)

// Optimizer interface: Step() updates parameters using their Grad.
type Optimizer interface {
	Step()
}

// ParamGroup holds a slice of parameter tensors (e.g. all weights and biases).
type ParamGroup struct {
	Params []*tensor.Tensor
}

// AdamWOptimizer wraps optim.AdamW and exposes Step.
type AdamWOptimizer struct {
	opt *optim.AdamW
}

// NewAdamW creates an AdamW optimizer for the given parameters.
func NewAdamW(params []*tensor.Tensor, lr, beta1, beta2, eps, weightDecay float64) *AdamWOptimizer {
	return &AdamWOptimizer{opt: optim.NewAdamW(params, lr, beta1, beta2, eps, weightDecay)}
}

// Step performs one update step.
func (o *AdamWOptimizer) Step() {
	o.opt.Step()
}
