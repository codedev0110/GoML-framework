package nn

import (
	"math"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// CrossEntropyLoss computes mean(-log(softmax(logits)[class])). logits [batch, C], target [batch] int64.
// Returns scalar. For backward, logits.Grad is filled with (softmax - one_hot) / batch.
func CrossEntropyLoss(logits, target *tensor.Tensor) (*tensor.Tensor, error) {
	if logits.DType != core.Float32 || len(logits.Shape) != 2 {
		return nil, nil
	}
	batch := logits.Shape[0]
	C := logits.Shape[1]
	be, err := backend.GetForDevice(logits.Storage.Device())
	if err != nil {
		return nil, err
	}
	// Numerically stable: max per row, then log_softmax, then nll.
	logitsF := logits.Float32()
	targetI := target.Int64()
	var loss float32
	for i := 0; i < batch; i++ {
		row := logitsF[i*C : (i+1)*C]
		maxV := row[0]
		for j := 1; j < C; j++ {
			if row[j] > maxV {
				maxV = row[j]
			}
		}
		var sumExp float32
		for j := 0; j < C; j++ {
			sumExp += float32(math.Exp(float64(row[j] - maxV)))
		}
		logSumExp := maxV + float32(math.Log(float64(sumExp)))
		t := targetI[i]
		if t < 0 || int(t) >= C {
			continue
		}
		loss -= (row[t] - logSumExp)
	}
	loss /= float32(batch)
	scalar, err := tensor.FromFloat32([]float32{loss}, 1)
	if err != nil {
		return nil, err
	}
	_ = be
	return scalar, nil
}
