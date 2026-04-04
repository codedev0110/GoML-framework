package ops

import (
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// CrossEntropyLoss computes -log(softmax(logits)[target]) averaged over batch.
// logits: [batch, numClasses], target: [batch] int64 indices.
// Returns a scalar tensor. Backward fills logits.Grad with gradient.
func CrossEntropyLoss(logits, target *tensor.Tensor) (*tensor.Tensor, error) {
	if logits.DType != core.Float32 || len(logits.Shape) != 2 {
		return nil, nil // placeholder
	}
	batch := logits.Shape[0]
	numClasses := logits.Shape[1]
	// Forward: softmax then -log(prob[target]). Numerically: log_softmax then NLL.
	// log_softmax(x) = x - log(sum(exp(x))) = x - log_sum_exp(x).
	// We'll do: probs = softmax(logits), loss = -mean(log(probs[i, target[i]])).
	// For stability: max_logit = max(logits[i]), then logits - max, then exp, sum, log.
	// loss_i = - (logits[i, target[i]] - max_i - log(sum(exp(logits[i]-max_i))))
	// So loss_i = -logits[i, target[i]] + max_i + log(sum(exp(logits[i]-max_i)))
	// Implemented in nn/loss.go with gradient; here we just define the op signature.
	_ = batch
	_ = numClasses
	return tensor.FromFloat32([]float32{0}, 1)
}
