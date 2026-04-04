package optim

import (
	"math"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// AdamW implements Adam with decoupled weight decay.
type AdamW struct {
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	t           int
	m           []backend.Storage // first moment
	v           []backend.Storage // second moment
}

// NewAdamW creates an AdamW optimizer. params are modified in place; they must have Grad set when Step() is called.
func NewAdamW(params []*tensor.Tensor, lr, beta1, beta2, eps, weightDecay float64) *AdamW {
	if eps == 0 {
		eps = 1e-8
	}
	be, _ := backend.GetForDevice(params[0].Storage.Device())
	m := make([]backend.Storage, len(params))
	v := make([]backend.Storage, len(params))
	for i, p := range params {
		byteLen := p.NumElements() * int(core.Float32.Size())
		m[i], _ = be.Alloc(byteLen)
		be.Fill(m[i], p.NumElements(), 0)
		v[i], _ = be.Alloc(byteLen)
		be.Fill(v[i], p.NumElements(), 0)
	}
	return &AdamW{
		params:      params,
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
		m:           m,
		v:           v,
	}
}

// Step performs one parameter update.
func (a *AdamW) Step() {
	a.t++
	for i, p := range a.params {
		if p.Grad == nil {
			continue
		}
		n := p.NumElements()
		grad := p.Grad.Float32()
		param := p.Float32()
		mF := backendStorageFloat32(a.m[i], n)
		vF := backendStorageFloat32(a.v[i], n)
		for j := 0; j < n; j++ {
			g := grad[j]
			// Weight decay
			param[j] -= float32(a.lr * a.weightDecay * float64(param[j]))
			mF[j] = float32(a.beta1)*mF[j] + float32(1-a.beta1)*g
			vF[j] = float32(a.beta2)*vF[j] + float32(1-a.beta2)*g*g
			mHat := mF[j] / float32(1-math.Pow(a.beta1, float64(a.t)))
			vHat := vF[j] / float32(1-math.Pow(a.beta2, float64(a.t)))
			param[j] -= float32(a.lr) * mHat / (float32(math.Sqrt(float64(vHat))) + float32(a.eps))
		}
	}
}

func backendStorageFloat32(s backend.Storage, n int) []float32 {
	b := s.Bytes()
	if b == nil || len(b) < n*4 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}
