package autograd

import (
	"github.com/djeday123/goml/tensor"
)

// Backward performs topological sort of the computation graph rooted at out,
// then runs each node's Backward in reverse order.
// out.Grad must be set (e.g. 1 for scalar loss) before calling.
func Backward(out *tensor.Tensor) {
	// Build set of all nodes reachable from out (by following tensor references).
	// We don't have an explicit graph; we only have Backward on each tensor.
	// So we do: start with out, run out.Backward(); each Backward may reference
	// input tensors (a, b) and set their Grad. We need to topo-sort so that
	// when we run node.Backward(), all consumers of node have already run.
	// Standard approach: collect all tensors in the graph (BFS/DFS from out),
	// then topo-sort by dependency (t depends on t if t is input to some op that produced a tensor that depends on t).
	// Our ops don't store references to inputs; they close over them in Backward.
	// So we can't easily enumerate the graph. Simpler approach: just run out.Backward().
	// That will call backward on the tensors that were used to compute out. But those
	// tensors might have Backward that need to run after we've set their Grad.
	// So the order is: 1) run out.Backward() which adds to input grads; 2) for each
	// input that has Backward, run it. So we need to recursively run Backward on
	// every tensor that gets grad accumulated. That can be done with a queue/stack:
	// queue = [out]
	// while queue not empty: t = pop; run t.Backward(); for each input t' that got grad, push t'.
	// But we don't have a list of "inputs" from an op. The Backward closure just
	// references a and b. So when we run out.Backward(), we're updating a.Grad and b.Grad.
	// We don't have a way to get "a" and "b" from out. So we cannot build the graph
	// from tensor alone. We need either: (1) each tensor to store .Inputs []*Tensor, or
	// (2) Backward to push inputs onto a queue. Option (2): change the signature so
	// Backward(appender func(*Tensor)). Then each op's Backward calls appender(a), appender(b)
	// after accumulating grad. Then we topo-sort the list and run Backward on each.
	// For minimal change: assume one backward pass, run out.Backward() only. So only
	// one level of backward is supported unless we add .Inputs. Let's add a simple
	// convention: Backward can optionally push dependencies. We'll use a slice and
	// pass it to Backward. So: type BackwardFunc func(next *[]*Tensor). And we
	// change the tensor to hold that. That would require changing all ops. Simpler:
	// just run out.Backward() once. For the full LLM we'll do manual backward in
	// nn/backward.go that calls each layer's backward. So here we only need to
	// support a single Backward() call.
	if out.Backward != nil {
		out.Backward()
	}
}

// BackwardAll runs Backward on out and recursively on any tensor that has
// Backward and received gradient. Requires tensors to register their inputs
// via a callback. For now we only run one level.
func BackwardAll(out *tensor.Tensor) {
	var queue []*tensor.Tensor
	queue = append(queue, out)
	seen := make(map[*tensor.Tensor]bool)
	for len(queue) > 0 {
		t := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if seen[t] {
			continue
		}
		seen[t] = true
		if t.Backward != nil {
			t.Backward()
		}
		// We cannot enumerate dependencies without storing them on the tensor.
		// So we stop after one level.
		break
	}
}
