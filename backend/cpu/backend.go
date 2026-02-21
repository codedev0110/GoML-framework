package cpu

import (
	"math"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

const tileSize = 32

type cpuBackend struct{}

func init() {
	backend.Register(&cpuBackend{})
}

func (b *cpuBackend) Name() string       { return "cpu" }
func (b *cpuBackend) DeviceType() backend.DeviceType { return backend.CPU }

func (b *cpuBackend) Alloc(byteLen int) (backend.Storage, error) {
	return Alloc(byteLen), nil
}

func (b *cpuBackend) Free(s backend.Storage) {
	if cs, ok := s.(*storage); ok {
		cs.Free()
	}
}

func (b *cpuBackend) Copy(dst, src backend.Storage, byteLen int) error {
	db := dst.(*storage).buf
	sb := src.(*storage).buf
	copy(db[:byteLen], sb[:byteLen])
	return nil
}

func (b *cpuBackend) ToDevice(d backend.Device, src backend.Storage) (backend.Storage, error) {
	if d.Type != backend.CPU {
		return nil, backend.ErrUnsupported
	}
	s := src.(*storage)
	out := make([]byte, len(s.buf))
	copy(out, s.buf)
	return &storage{buf: out, dev: d}, nil
}

func floatSlice(s backend.Storage, n int) []float32 {
	if n == 0 {
		return nil
	}
	b := s.(*storage).buf
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}

func (b *cpuBackend) Neg(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = -x[i]
	}
	return nil
}

func (b *cpuBackend) Abs(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(math.Abs(float64(x[i])))
	}
	return nil
}

func (b *cpuBackend) Exp(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(math.Exp(float64(x[i])))
	}
	return nil
}

func (b *cpuBackend) Log(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(math.Log(float64(x[i])))
	}
	return nil
}

func (b *cpuBackend) Sqrt(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(math.Sqrt(float64(x[i])))
	}
	return nil
}

func (b *cpuBackend) Tanh(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(math.Tanh(float64(x[i])))
	}
	return nil
}

func (b *cpuBackend) Relu(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		if x[i] > 0 {
			d[i] = x[i]
		} else {
			d[i] = 0
		}
	}
	return nil
}

func (b *cpuBackend) Gelu(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		xf := float64(x[i])
		d[i] = float32(0.5 * xf * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(xf+0.044715*xf*xf*xf))))
	}
	return nil
}

func (b *cpuBackend) Sigmoid(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = float32(1 / (1 + math.Exp(-float64(x[i]))))
	}
	return nil
}

func (b *cpuBackend) Silu(dst, src backend.Storage, nElems int) error {
	d := floatSlice(dst, nElems)
	x := floatSlice(src, nElems)
	for i := range d {
		d[i] = x[i] * float32(1/(1+math.Exp(-float64(x[i]))))
	}
	return nil
}

// broadcastIter: for each linear out index, compute linear indices into a and b (NumPy broadcast).
func broadcastIter(outShape core.Shape, aShape, bShape core.Shape, aStrides, bStrides core.Strides,
	outStrides core.Strides) (nOut int, getIndices func(outLinear int) (aIdx, bIdx int)) {
	nOut = 1
	for _, d := range outShape {
		nOut *= d
	}
	nd := len(outShape)
	aPad := nd - len(aShape)
	bPad := nd - len(bShape)
	getIndices = func(outLinear int) (aIdx, bIdx int) {
		rem := outLinear
		idx := make([]int, nd)
		for i := nd - 1; i >= 0; i-- {
			idx[i] = rem % outShape[i]
			rem /= outShape[i]
		}
		for i := 0; i < nd; i++ {
			if i >= aPad {
				aIdx += idx[i] * (aStrides[i-aPad] / 4)
			}
			if i >= bPad {
				bIdx += idx[i] * (bStrides[i-bPad] / 4)
			}
		}
		return aIdx, bIdx
	}
	return nOut, getIndices
}

func outStrides(shape core.Shape) core.Strides {
	return core.ContiguousStrides(shape, 4)
}

func (b *cpuBackend) Add(dst, a, b backend.Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error {
	outStr := outStrides(outShape)
	n, get := broadcastIter(outShape, aShape, bShape, aStrides, bStrides, outStr)
	da := floatSlice(dst, n)
	pa := floatSlice(a, aShape.NumElements())
	pb := floatSlice(b, bShape.NumElements())
	for i := 0; i < n; i++ {
		ai, bi := get(i)
		da[i] = pa[ai] + pb[bi]
	}
	return nil
}

func (b *cpuBackend) Sub(dst, a, b backend.Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error {
	outStr := outStrides(outShape)
	n, get := broadcastIter(outShape, aShape, bShape, aStrides, bStrides, outStr)
	da := floatSlice(dst, n)
	pa := floatSlice(a, aShape.NumElements())
	pb := floatSlice(b, bShape.NumElements())
	for i := 0; i < n; i++ {
		ai, bi := get(i)
		da[i] = pa[ai] - pb[bi]
	}
	return nil
}

func (b *cpuBackend) Mul(dst, a, b backend.Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error {
	outStr := outStrides(outShape)
	n, get := broadcastIter(outShape, aShape, bShape, aStrides, bStrides, outStr)
	da := floatSlice(dst, n)
	pa := floatSlice(a, aShape.NumElements())
	pb := floatSlice(b, bShape.NumElements())
	for i := 0; i < n; i++ {
		ai, bi := get(i)
		da[i] = pa[ai] * pb[bi]
	}
	return nil
}

func (b *cpuBackend) Div(dst, a, b backend.Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error {
	outStr := outStrides(outShape)
	n, get := broadcastIter(outShape, aShape, bShape, aStrides, bStrides, outStr)
	da := floatSlice(dst, n)
	pa := floatSlice(a, aShape.NumElements())
	pb := floatSlice(b, bShape.NumElements())
	for i := 0; i < n; i++ {
		ai, bi := get(i)
		da[i] = pa[ai] / pb[bi]
	}
	return nil
}

func (b *cpuBackend) Sum(dst, src backend.Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error {
	// axis -1 = reduce all
	srcF := floatSlice(src, srcShape.NumElements())
	if axis < 0 {
		axis = len(srcShape) + axis
	}
	var sum float32
	if axis < 0 || len(srcShape) == 0 {
		for _, v := range srcF {
			sum += v
		}
		dstF := floatSlice(dst, 1)
		dstF[0] = sum
		return nil
	}
	// Reduce along axis: output shape = drop axis (or 1 if keepDim)
	before := 1
	for i := 0; i < axis; i++ {
		before *= srcShape[i]
	}
	after := 1
	for i := axis + 1; i < len(srcShape); i++ {
		after *= srcShape[i]
	}
	dimSize := srcShape[axis]
	strideAxis := srcStrides[axis] / 4
	dstF := floatSlice(dst, before*after)
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			var s float32
			for k := 0; k < dimSize; k++ {
				// flat index into src
				srcIdx := i*after*dimSize + k*after + j
				// more general: need to compute from strides
				off := 0
				ii, jj := i, j
				for d := 0; d < axis; d++ {
					off += (ii % srcShape[d]) * (srcStrides[d] / 4)
					ii /= srcShape[d]
				}
				off += k * strideAxis
				for d := axis + 1; d < len(srcShape); d++ {
					off += (jj % srcShape[d]) * (srcStrides[d] / 4)
					jj /= srcShape[d]
				}
				s += srcF[off]
			}
			dstF[i*after+j] = s
		}
	}
	return nil
}

func (b *cpuBackend) Max(dst, src backend.Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error {
	srcF := floatSlice(src, srcShape.NumElements())
	if axis < 0 {
		axis = len(srcShape) + axis
	}
	if axis < 0 || len(srcShape) == 0 {
		maxV := srcF[0]
		for _, v := range srcF[1:] {
			if v > maxV {
				maxV = v
			}
		}
		floatSlice(dst, 1)[0] = maxV
		return nil
	}
	before := 1
	for i := 0; i < axis; i++ {
		before *= srcShape[i]
	}
	after := 1
	for i := axis + 1; i < len(srcShape); i++ {
		after *= srcShape[i]
	}
	dimSize := srcShape[axis]
	strideAxis := srcStrides[axis] / 4
	dstF := floatSlice(dst, before*after)
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			maxV := float32(-1e9)
			for k := 0; k < dimSize; k++ {
				off := i*after*dimSize + k*after + j
				if axis != 0 {
					off = 0
					ii, jj := i, j
					for d := 0; d < axis; d++ {
						off += (ii % srcShape[d]) * (srcStrides[d] / 4)
						ii /= srcShape[d]
					}
					off += k * strideAxis
					for d := axis + 1; d < len(srcShape); d++ {
						off += (jj % srcShape[d]) * (srcStrides[d] / 4)
						jj /= srcShape[d]
					}
				}
				if srcF[off] > maxV {
					maxV = srcF[off]
				}
			}
			dstF[i*after+j] = maxV
		}
	}
	return nil
}

func (b *cpuBackend) Mean(dst, src backend.Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error {
	if err := b.Sum(dst, src, srcShape, srcStrides, axis, keepDim); err != nil {
		return err
	}
	dimSize := 1
	if axis >= 0 && axis < len(srcShape) {
		dimSize = srcShape[axis]
	} else {
		dimSize = srcShape.NumElements()
	}
	outSize := srcShape.NumElements() / dimSize
	dstF := floatSlice(dst, outSize)
	for i := range dstF {
		dstF[i] /= float32(dimSize)
	}
	return nil
}

func (b *cpuBackend) MatMul(dst, a, b backend.Storage, batchSize, M, N, K int) error {
	d := floatSlice(dst, batchSize*M*N)
	pa := floatSlice(a, batchSize*M*K)
	pb := floatSlice(b, batchSize*K*N)
	for batch := 0; batch < batchSize; batch++ {
		aBase := batch * M * K
		bBase := batch * K * N
		cBase := batch * M * N
		// Tiled matmul
		for i0 := 0; i0 < M; i0 += tileSize {
			for k0 := 0; k0 < K; k0 += tileSize {
				for j0 := 0; j0 < N; j0 += tileSize {
					iEnd := i0 + tileSize
					if iEnd > M {
						iEnd = M
					}
					kEnd := k0 + tileSize
					if kEnd > K {
						kEnd = K
					}
					jEnd := j0 + tileSize
					if jEnd > N {
						jEnd = N
					}
					for i := i0; i < iEnd; i++ {
						for k := k0; k < kEnd; k++ {
							aik := pa[aBase+i*K+k]
							for j := j0; j < jEnd; j++ {
								d[cBase+i*N+j] += aik * pb[bBase+k*N+j]
							}
						}
					}
				}
			}
		}
	}
	return nil
}

func (b *cpuBackend) Softmax(dst, src backend.Storage, shape core.Shape, strides core.Strides) error {
	n := shape.NumElements()
	if n == 0 {
		return nil
	}
	lastDim := shape[len(shape)-1]
	outer := n / lastDim
	srcF := floatSlice(src, n)
	dstF := floatSlice(dst, n)
	lastStride := strides[len(strides)-1] / 4
	for o := 0; o < outer; o++ {
		base := o * lastDim
		// max for numerical stability
		maxV := srcF[base]
		for i := 1; i < lastDim; i++ {
			if srcF[base+i] > maxV {
				maxV = srcF[base+i]
			}
		}
		var sum float32
		for i := 0; i < lastDim; i++ {
			v := float32(math.Exp(float64(srcF[base+i] - maxV)))
			dstF[base+i] = v
			sum += v
		}
		for i := 0; i < lastDim; i++ {
			dstF[base+i] /= sum
		}
	}
	_ = lastStride
	return nil
}

func (b *cpuBackend) LayerNorm(dst, x, gamma, beta, mean, var_ backend.Storage, shape core.Shape, strides core.Strides, eps float32) error {
	n := shape.NumElements()
	lastDim := shape[len(shape)-1]
	outer := n / lastDim
	xF := floatSlice(x, n)
	dstF := floatSlice(dst, n)
	gammaF := floatSlice(gamma, lastDim)
	betaF := floatSlice(beta, lastDim)
	meanF := floatSlice(mean, outer)
	varF := floatSlice(var_, outer)
	lastStride := strides[len(strides)-1] / 4
	for o := 0; o < outer; o++ {
		base := o * lastDim
		var sum float32
		for i := 0; i < lastDim; i++ {
			sum += xF[base+i]
		}
		m := sum / float32(lastDim)
		meanF[o] = m
		var s float32
		for i := 0; i < lastDim; i++ {
			d := xF[base+i] - m
			s += d * d
		}
		v := s/float32(lastDim) + eps
		varF[o] = v
		inv := float32(1 / math.Sqrt(float64(v)))
		for i := 0; i < lastDim; i++ {
			dstF[base+i] = (xF[base+i]-m)*inv*gammaF[i] + betaF[i]
		}
	}
	_ = lastStride
	return nil
}

func (b *cpuBackend) Embedding(dst, table backend.Storage, indices backend.Storage, tableRows, tableCols int) error {
	idxB := indices.(*storage).buf
	idx := unsafe.Slice((*int64)(unsafe.Pointer(&idxB[0])), len(idxB)/8)
	outF := floatSlice(dst, len(idx)*tableCols)
	tabF := floatSlice(table, tableRows*tableCols)
	for i, pos := range idx {
		if pos < 0 || int(pos) >= tableRows {
			continue
		}
		row := int(pos) * tableCols
		for j := 0; j < tableCols; j++ {
			outF[i*tableCols+j] = tabF[row+j]
		}
	}
	return nil
}

func (b *cpuBackend) RoPE(dst, x backend.Storage, shape core.Shape, strides core.Strides, base float64, startPos, seqLen int) error {
	// shape is [..., seq, head_dim]. Last dim is head_dim (must be even).
	n := shape.NumElements()
	headDim := shape[len(shape)-1]
	if headDim%2 != 0 {
		return backend.ErrUnsupported
	}
	half := headDim / 2
	seqDim := shape[len(shape)-2]
	xF := floatSlice(x, n)
	dstF := floatSlice(dst, n)
	copy(dstF, xF)
	// inv_freq[i] = 1 / (base^(2i/D))
	invFreq := make([]float64, half)
	for i := range invFreq {
		invFreq[i] = 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
	}
	// per (batch..., seq, head_dim): apply rotation per position
	rowSize := headDim
	rowsPerSeq := n / (seqDim * rowSize)
	for b := 0; b < rowsPerSeq; b++ {
		for s := 0; s < seqLen && (startPos+s) < seqDim; s++ {
			pos := startPos + s
			theta := float64(pos) * invFreq[0]
			baseOff := (b*seqDim+pos)*rowSize + 0
			for i := 0; i < half; i++ {
				theta := float64(pos) * invFreq[i]
				cos := float32(math.Cos(theta))
				sin := float32(math.Sin(theta))
				x0 := dstF[baseOff+i]
				x1 := dstF[baseOff+half+i]
				dstF[baseOff+i] = x0*cos - x1*sin
				dstF[baseOff+half+i] = x0*sin + x1*cos
			}
		}
	}
	return nil
}

func (b *cpuBackend) ScaledDotProductAttention(dst, q, k, v backend.Storage, batch, heads, seq, headDim int, causal bool) error {
	scale := 1 / float32(math.Sqrt(float64(headDim)))
	qF := floatSlice(q, batch*heads*seq*headDim)
	kF := floatSlice(k, batch*heads*seq*headDim)
	vF := floatSlice(v, batch*heads*seq*headDim)
	dstF := floatSlice(dst, batch*heads*seq*headDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			qBase := (b*heads+h)*seq*headDim
			kBase := (b*heads+h)*seq*headDim
			vBase := (b*heads+h)*seq*headDim
			// scores = Q @ K^T  [seq, seq]
			scores := make([]float32, seq*seq)
			for i := 0; i < seq; i++ {
				for j := 0; j < seq; j++ {
					if causal && j > i {
						scores[i*seq+j] = -1e9
						continue
					}
					var dot float32
					for d := 0; d < headDim; d++ {
						dot += qF[qBase+i*headDim+d] * kF[kBase+j*headDim+d]
					}
					scores[i*seq+j] = dot * scale
				}
			}
			// softmax over j
			for i := 0; i < seq; i++ {
				maxV := scores[i*seq]
				for j := 1; j < seq; j++ {
					if scores[i*seq+j] > maxV {
						maxV = scores[i*seq+j]
					}
				}
				var sum float32
				for j := 0; j < seq; j++ {
					scores[i*seq+j] = float32(math.Exp(float64(scores[i*seq+j] - maxV)))
					sum += scores[i*seq+j]
				}
				for j := 0; j < seq; j++ {
					scores[i*seq+j] /= sum
				}
			}
			// out = scores @ V
			outBase := (b*heads+h)*seq*headDim
			for i := 0; i < seq; i++ {
				for d := 0; d < headDim; d++ {
					var s float32
					for j := 0; j < seq; j++ {
						s += scores[i*seq+j] * vF[vBase+j*headDim+d]
					}
					dstF[outBase+i*headDim+d] = s
				}
			}
		}
	}
	return nil
}

func (b *cpuBackend) Fill(dst backend.Storage, nElems int, value float32) error {
	d := floatSlice(dst, nElems)
	for i := range d {
		d[i] = value
	}
	return nil
}

func (b *cpuBackend) Arange(dst backend.Storage, nElems int, start, step float32) error {
	d := floatSlice(dst, nElems)
	for i := range d {
		d[i] = start + float32(i)*step
	}
	return nil
}

func (b *cpuBackend) Where(dst, cond, a, b backend.Storage, nElems int) error {
	// cond as float32: nonzero -> a else b
	d := floatSlice(dst, nElems)
	c := floatSlice(cond, nElems)
	pa := floatSlice(a, nElems)
	pb := floatSlice(b, nElems)
	for i := range d {
		if c[i] != 0 {
			d[i] = pa[i]
		} else {
			d[i] = pb[i]
		}
	}
	return nil
}
