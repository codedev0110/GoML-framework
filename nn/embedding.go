package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Embedding is a lookup table: out = table[indices]. Table [numEmbeddings, embedDim].
type Embedding struct {
	Table *tensor.Tensor // [numEmbeddings, embedDim]
}

// NewEmbedding wraps a table tensor.
func NewEmbedding(table *tensor.Tensor) (*Embedding, error) {
	if len(table.Shape) != 2 {
		return nil, fmt.Errorf("embedding table must be 2D")
	}
	return &Embedding{Table: table}, nil
}

// Forward returns table[indices]. indices: int64 tensor shape [..., ]; out: [..., embedDim].
func (e *Embedding) Forward(indices *tensor.Tensor) (*tensor.Tensor, error) {
	if indices.DType != core.Int64 {
		return nil, fmt.Errorf("indices must be int64")
	}
	numEmbeddings := e.Table.Shape[0]
	embedDim := e.Table.Shape[1]
	indexCount := indices.NumElements()
	be, err := backend.GetForDevice(e.Table.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, err := be.Alloc(indexCount * embedDim * 4)
	if err != nil {
		return nil, err
	}
	if err := be.Embedding(outStorage, e.Table.Storage, indices.Storage, numEmbeddings, embedDim); err != nil {
		outStorage.Free()
		return nil, err
	}
	outShape := make(core.Shape, len(indices.Shape)+1)
	copy(outShape, indices.Shape)
	outShape[len(outShape)-1] = embedDim
	strides := core.ContiguousStrides(outShape, 4)
	out := tensor.New(outStorage, outShape, strides, core.Float32)

	// Store for backward
	out.Backward = func() {
		e.BackwardFunction(indices, out)
	}

	return out, nil
}

// BackwardFunction accumulates gradients to the embedding table.
// Expects out.Grad to be set (gradient with respect to output).
// We scatter gradients from output back to table[indices].
func (e *Embedding) BackwardFunction(indices, out *tensor.Tensor) {
	if out.Grad == nil {
		return
	}

	be, err := backend.GetForDevice(e.Table.Storage.Device())
	if err != nil {
		return
	}

	embedDim := e.Table.Shape[1]
	numEmbeddings := e.Table.Shape[0]

	// Initialize table gradient if not present
	if e.Table.Grad == nil {
		tableGradSize := numEmbeddings * embedDim * 4
		tableGradStorage, _ := be.Alloc(tableGradSize)
		be.Fill(tableGradStorage, numEmbeddings*embedDim, 0)
		e.Table.Grad = tensor.New(tableGradStorage, e.Table.Shape,
			core.ContiguousStrides(e.Table.Shape, 4), core.Float32)
	}

	// Scatter gradients: for each index in indices, add out.Grad to table.Grad[index]
	indicesI64 := indices.Int64()
	outGradF := out.Grad.Float32()
	tableGradF := e.Table.Grad.Float32()

	indexCount := indices.NumElements()
	for idx := 0; idx < indexCount; idx++ {
		embIdx := int(indicesI64[idx])
		if embIdx >= 0 && embIdx < numEmbeddings {
			for d := 0; d < embedDim; d++ {
				tableGradF[embIdx*embedDim+d] += outGradF[idx*embedDim+d]
			}
		}
	}
}
