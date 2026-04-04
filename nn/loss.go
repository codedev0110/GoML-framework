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
	if logits.DType() != core.Float32 || logits.NDim() != 2 {
		return nil, nil
	}
	batch := logits.Shape()[0]
	C := logits.Shape()[1]
	be, err := backend.GetForDevice(logits.Storage().Device())
	if err != nil {
		return nil, err
	}
	logitsF := logits.ToFloat32Slice()
	targetI := target.ToInt64Slice()

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

	// Set grad function: when called, fills logits.Grad = (softmax - one_hot) / batch
	scalar.SetGradFn(func() {
		// Allocate grad for logits
		gradStorage, err2 := be.Alloc(batch * C * int(core.Float32.Size()))
		if err2 != nil {
			return
		}
		lg := tensor.NewTensor(gradStorage, logits.Shape(), core.Float32)
		lg.SetRequiresGrad(true)
		logits.SetGrad(lg)
		gradF := logits.Grad().ToFloat32Slice()
		for i := 0; i < batch; i++ {
			t := targetI[i]
			for j := 0; j < C; j++ {
				gradF[i*C+j] = softmax[i*C+j] / float32(batch)
			}
			if t >= 0 && int(t) < C {
				gradF[i*C+int(t)] -= 1.0 / float32(batch)
			}
		}
	})

	return scalar, nil
}
