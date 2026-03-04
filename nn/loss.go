package nn

import (
	"math"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// CrossEntropyLoss computes mean(-log(softmax(logits)[class])). logits [batch, C], target [batch] int64.
// Returns scalar. Backward fills logits.Grad with (softmax - one_hot) / batch.
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
	logitsF := logits.Float32()
	targetI := target.Int64()

	// Pre-compute softmax for each row (numerically stable)
	softmax := make([]float32, batch*C)
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
			softmax[i*C+j] = float32(math.Exp(float64(row[j] - maxV)))
			sumExp += softmax[i*C+j]
		}
		for j := 0; j < C; j++ {
			softmax[i*C+j] /= sumExp
		}
		t := targetI[i]
		if t < 0 || int(t) >= C {
			continue
		}
		logSumExp := maxV + float32(math.Log(float64(sumExp)))
		loss -= (row[t] - logSumExp)
	}
	loss /= float32(batch)

	scalar, err := tensor.FromFloat32([]float32{loss}, 1)
	if err != nil {
		return nil, err
	}

	// Set Backward closure: when called, fills logits.Grad = (softmax - one_hot) / batch
	scalar.Backward = func() {
		// Allocate grad for logits
		gradStorage, err2 := be.Alloc(batch * C * 4)
		if err2 != nil {
			return
		}
		logits.Grad = tensor.New(gradStorage, logits.Shape, logits.Strides, core.Float32)
		gradF := logits.Grad.Float32()
		for i := 0; i < batch; i++ {
			t := targetI[i]
			for j := 0; j < C; j++ {
				gradF[i*C+j] = softmax[i*C+j] / float32(batch)
			}
			if t >= 0 && int(t) < C {
				gradF[i*C+int(t)] -= 1.0 / float32(batch)
			}
		}
	}

	return scalar, nil
}
