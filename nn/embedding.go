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
	return tensor.New(outStorage, outShape, strides, core.Float32), nil
}
