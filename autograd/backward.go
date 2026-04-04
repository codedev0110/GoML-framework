package autograd

import (
	"github.com/djeday123/goml/tensor"
)

// Backward initiates a backward pass from `out`.  The caller must assign a
// gradient to `out` before invoking this function (e.g. 1 for a scalar loss).
//
// The implementation here is intentionally simple: it merely invokes the
// GradFn stored on each tensor.  In this toy framework the gradient
// functions themselves close over any inputs and perform the appropriate
// accumulation, so an explicit graph traversal is unnecessary.  More complete
// frameworks build a dependency graph and execute nodes in topologically
// sorted order.
func Backward(out *tensor.Tensor) {
	if fn := out.GradFn(); fn != nil {
		fn()
	}
}

// BackwardAll is a compatibility helper; it currently behaves identically to
// Backward.  A future implementation could recursively traverse the graph.
func BackwardAll(out *tensor.Tensor) {
	Backward(out)
}
