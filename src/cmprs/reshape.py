from cmprs.transform import Transform, FOut


class FlattenTransform(Transform):
    def forward(self, x):
        return FOut(x.view(-1), x.shape)

    def backward(self, tx, original_shape):
        return tx.view(original_shape)


flatten = FlattenTransform()
