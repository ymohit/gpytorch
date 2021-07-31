import torch
from numpy import prod
from torch import Size, Tensor

from ..utils.getitem import _is_noop_index, _noop_index
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor


def _sparse_matmul(sparse_tensor, rhs):
    # TODO: test for rhs with both 2-D and 3-D shapes, i.e, * X * and b X * X * .
    if sparse_tensor.ndim <= 2:
        if rhs.ndim == 1:
            return torch.mv(sparse_tensor, rhs)
        elif rhs.ndim == 2:
            return torch.sparse.mm(sparse_tensor, rhs)
        elif rhs.ndim == 3:
            # a batched soln from https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
            # this works for non batched mvms
            batch_size = rhs.shape[0]
            # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
            vectors = rhs.transpose(0, 1).reshape(sparse_tensor.shape[1], -1)
            # A matrix-matrix product is a batched matrix-vector product of the columns.
            # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
            return (
                torch.sparse.mm(sparse_tensor, vectors).reshape(sparse_tensor.shape[-1], batch_size, -1).transpose(1, 0)
            )
    else:
        # we need to compute a direct sum (b, m, n) -> (bm, bn) to flatten the sparse tensor to have two
        # dimensions
        sparse_tensor = sparse_tensor.coalesce()
        indices, values = sparse_tensor.indices(), sparse_tensor.values()
        tsr_shape = list(sparse_tensor.shape)
        updated_size = [tsr_shape[0] * tsr_shape[1], tsr_shape[0] * tsr_shape[2]]
        new_sparse_tensor = torch.sparse_coo_tensor(
            indices=indices[0] * indices[1:], values=values, size=updated_size, device=indices.device
        ).coalesce()
        if rhs.ndim < 3:
            new_rhs = torch.cat([rhs for _ in range(tsr_shape[0])], dim=0)
        else:
            new_rhs = torch.cat([rhs[i] for i in range(tsr_shape[0])], dim=0)

        direct_result = _sparse_matmul(new_sparse_tensor, new_rhs.contiguous())
        result = direct_result.reshape(tsr_shape[0], tsr_shape[1], -1)
        return result


# def _sparse_add(sparse_tensor, rhs):
#     # TODO: test for rhs with both 2-D and 3-D shapes, i.e, * X * and b X * X * .
#     if sparse_tensor.ndim <= 2:
#         if rhs.ndim <= 2:
#             return rhs + sparse_tensor
#         else:
#             # TODO: a similar to case to that above
#             # raise NotImplementedError("Batched rhs addition is not implemented yet.")
#             sparse_tensor = sparse_tensor.coalesce()
#             indices, values = sparse_tensor.indices(), sparse_tensor.values()
#             tsr_shape = list(sparse_tensor.shape)
#             updated_size = [rhs.shape[0] * tsr_shape[0], rhs.shape[0] * tsr_shape[1]]
#             # the new sparse tensor is now (bm, bn)
#             new_indices = torch.cat([idx * indices for idx in range(rhs.shape[0])], dim=1)
#             new_values = torch.cat([values for _ in range(rhs.shape[0])], dim=0)
#             new_sparse_tensor = torch.sparse_coo_tensor(
#                 indices=new_indices, values=new_values, size=updated_size, device=indices.device
#             ).coalesce()
#             # we now need to reshape the rhs from (b, m, n) -> (bm, bn)
#             new_rhs = torch.cat([rhs[i] for i in range(rhs.shape[0])], dim=0)
#             # we need to tile this, TODO: replace the tiling with zeros
#             new_rhs = torch.cat([new_rhs for _ in range(rhs.shape[0])], dim=1)
#             direct_result = _sparse_add(new_sparse_tensor, new_rhs.contiguous())
#             # now we need to matmul with stacked identities
#             stacked_eye = torch.cat([torch.eye(rhs.shape[1]) for _ in range(rhs.shape[0])], dim=0)
#             direct_result = direct_result.matmul(stacked_eye)

#             result = direct_result.reshape(rhs.shape[0], tsr_shape[1], -1)
#             return result
#     else:
#         # we need to compute a direct sum (b, m, n) -> (bm, bn) to flatten the sparse tensor to have two
#         # dimensions
#         sparse_tensor = sparse_tensor.coalesce()
#         indices, values = sparse_tensor.indices(), sparse_tensor.values()
#         tsr_shape = list(sparse_tensor.shape)
#         updated_size = [tsr_shape[0] * tsr_shape[1], tsr_shape[0], tsr_shape[2]]
#         new_sparse_tensor = torch.sparse_coo_tensor(
#             indices=indices[0] * indices[1:], values=values, size=updated_size, device=indices.device
#         ).coalesce()
#         if rhs.ndim < 3:
#             new_rhs = torch.cat([rhs for _ in range(tsr_shape[0])], dim=0)
#         else:
#             print(rhs.shape, tsr_shape)
#             new_rhs = torch.cat([rhs[i] for i in range(tsr_shape[0])], dim=0)

#         print(new_sparse_tensor.shape, new_rhs.shape)
#         direct_result = _sparse_add(new_sparse_tensor, new_rhs.contiguous())
#         result = direct_result.reshape(tsr_shape[0], tsr_shape[1], -1)
#         return result


class SparseLazyTensor(LazyTensor):
    def __init__(self, indices: Tensor, values: Tensor, sparse_size: Size):
        """
        Sparse Lazy Tensor. Lazify torch.sparse_coo_tensor and supports arbitrary batch sizes.
        Args:
            :param indices: `b1 x ... x bk x ndim x nse` `tensor` containing indices of a `b1 x ... x bk`-sized batch
                    of sparse matrices with `sparse_size`.
            :param values: `b1 x ... x bk x nse` `tensor` containing values of a `b1 x ... x bk`-sized batch
                    of sparse matrices with `sparse_size`.
            :param sparse_size: `tensor` containing shape of non-batched dimensions of sparse matrices.

        TODO: revisit this as it seems to me that ndim=2 is sufficient for most cases.
        """
        super().__init__(indices, values, torch.tensor(sparse_size))

        # Local variable to keep batch shape as batch dimensions are squeezed in _tensor for efficiency.
        # self._batch_shape = indices.shape[:-2]
        self.sparse_size = torch.Size(sparse_size)
        self._batch_shape = self.sparse_size[:-2]

        ndim, nse = indices.shape[-2], indices.shape[-1]

        self.ndim = ndim  # dimension of the sparse matrices
        self.nse = nse  # number of specified elements

        tensor = torch.sparse_coo_tensor(
            indices=indices, values=values, size=self.sparse_size, device=indices.device, dtype=values.dtype
        )

        self._tensor = tensor.coalesce()

    @property
    def dtype(self):
        return self._tensor.dtype

    def to_dense(self):
        return self._tensor.to_dense().reshape(*self.size())

    def _size(self):
        return torch.Size(self._batch_shape + self._tensor.shape[-2:])

    def compute_effective_batch_index(self, *batch_indices):
        shifted_shapes = (*self.batch_shape[:-1], 1)[1:]
        return sum(bs * bi for bs, bi in zip(shifted_shapes, batch_indices))

    def _transpose_nonbatch(self):
        # TODO: this is implemented assuming ndim is 2.
        tensor_indices = self._tensor._indices().clone()
        new_indices = torch.zeros_like(tensor_indices)
        new_indices[..., 0, :] = new_indices[..., 1, :]
        new_indices[..., 1, :] = new_indices[..., 0, :]
        return self.__class__(indices=new_indices, values=self._tensor._values(), sparse_size=self.sparse_size)

    @cached
    def evaluate(self):
        return self._tensor.to_dense().reshape(self.shape)

    def _matmul(self, rhs: Tensor) -> Tensor:
        return _sparse_matmul(self._tensor, rhs)

    def diag(self):
        # TODO: there is a more efficient algorithm by searching for matches in the indices
        return self.to_dense().diag()

    def matmul(self, tensor):
        return self._matmul(rhs=tensor)

    def _mul_constant(self, constant):

        if self.ndimension() > 2:
            ndim, nse = self._tensor.indices().shape[-2:]
            return self.__class__(
                indices=self._tensor._indices().reshape(*self.batch_shape, ndim, nse),
                values=constant * self._tensor._values.reshape(*self.batch_shape, nse),
                sparse_size=self.sparse_size,
            )
        else:
            return self.__class__(
                indices=self._tensor._indices(), values=constant * self._tensor._values(), sparse_size=self.sparse_size,
            )

    def _t_matmul(self, rhs):
        return self._transpose_nonbatch().matmul(rhs)

    def _expand_batch(self, batch_shape):

        if not self._tensor.is_coalesced():
            self._tensor = self._tensor.coalesce()

        # TODO: probably want to store indices, values as a property to avoid repeated fn calls
        # we now have (batch_shape) * nz non zero indices
        current_indices = self._tensor.indices()
        starting_values = self._tensor.values()

        expanded_values = starting_values.repeat_interleave(prod(batch_shape))
        # TODO: is there something more efficient than a for loop?
        ndims = len(batch_shape)
        for dim in range(ndims):
            rg = torch.arange(batch_shape[dim]).to(starting_values)
            interleaved_rg = rg.repeat_interleave(starting_values.shape[-1])
            current_indices = torch.cat([current_indices] * batch_shape[dim], dim=-1)
            current_indices = torch.cat([interleaved_rg.unsqueeze(0), current_indices], dim=0)

        expanded_shape = torch.Size([*batch_shape, *self.sparse_size])
        return self.__class__(indices=current_indices, values=expanded_values, sparse_size=expanded_shape)

    def _getitem(self, row_index, col_index, *batch_indices):
        if len(self.batch_shape) > 0:
            effective_batch_index = self.compute_effective_batch_index(batch_indices)
            return self._tensor[(effective_batch_index, row_index, col_index)]
        else:
            print("tensor: ", self._tensor, type(row_index), col_index)
            print(
                "done --> ",
                row_index is _noop_index,
                row_index is _noop_index,
                _is_noop_index(row_index),
                _is_noop_index(col_index),
            )
            return self._tensor[row_index, col_index]

    # def _get_indices(self, row_index, col_index, *batch_indices):
    #     if len(self.batch_shape) > 0:
    #         effective_batch_index = self.compute_effective_batch_index(batch_indices)
    #         return self._tensor[(effective_batch_index, row_index, col_index)]
    #     else:
    #         print("tensor: ", self._tensor, self._tensor[0, 1], row_index, col_index)
    #         return self._tensor[row_index, col_index]

    def _unsqueeze_batch(self, dim):
        new_batch_shape = torch.Size((*self._batch_shape[:dim], 1, *self._batch_shape[dim:]))
        return self.__class__(
            indices=self._tensor.indices().reshape(*new_batch_shape, self.ndim, self.nse),
            values=self._tensor.values().reshape(*new_batch_shape, self.nse),
            sparse_size=self.sparse_size,
        )

    def __add__(self, other):
        # if isinstance(other, SparseLazyTensor):
        #     new_sparse_lazy_tensor = copy.deepcopy(self)
        #     new_sparse_lazy_tensor._tensor = new_sparse_lazy_tensor._tensor + other._tensor
        #     return new_sparse_lazy_tensor
        if isinstance(other, LazyTensor) or isinstance(other, SparseLazyTensor):
            return super(SparseLazyTensor, self).__add__(other)
        if isinstance(other, torch.Tensor):
            # return NonLazyTensor(_sparse_add(self._tensor, other))
            # TODO: use sparse addition for speed gains
            return NonLazyTensor(self._tensor.to_dense() + other)

    def _sum_batch(self, dim):
        indices = self._tensor.indices().reshape(self.batch_shape, self.ndim, self.nse)
        values = self._tensor.values().reshape(self.batch_shape, self.nse)

        indices_splits = torch.split(indices, indices.shape[dim], dim)
        values_splits = torch.split(values, indices.shape[dim], dim)

        return sum(
            [
                self.__class__(indices=indices_split, values=values_split, sparse_size=self.sparse_size)
                for indices_split, values_split in zip(indices_splits, values_splits)
            ]
        )

    def sum(self, dim=None):
        return self._tensor.sum(dim)

    def _permute_batch(self, *dims):
        indices = self._tensor.indices().reshape(self.batch_shape, self.ndim, self.nse)
        values = self._tensor.values().reshape(self.batch_shape, self.nse)
        indices = indices.permute(*dims, -2, -1)
        values = values.permute(*dims, -1)
        return self.__class__(indices=indices, values=values, sparse_size=self.sparse_size)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # TODO: keep this as a reminder to revisit
        return super()._quad_form_derivative(left_vecs=left_vecs, right_vecs=right_vecs)

    def _cholesky_solve(self, rhs, upper: bool = False):
        raise NotImplementedError
