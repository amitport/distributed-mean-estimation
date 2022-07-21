from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Any, Optional


@dataclass
class FOut:
    """ The return type of the Transform forward method """
    tx: Any
    tcontext: Optional[Any] = None
    bitrate: Optional[float] = None


@dataclass
class ROut:
    rx: Any
    bitrate: Optional[float]


class Transform(ABC):

    @abstractmethod
    def forward(self, x) -> FOut:
        """
        :param x: An object to transform
        :return: FOut Transformed x and additional context, will be used as an input for '_backward'
                      the optional 'bitrate' propagates as a metric
        """

    # noinspection PyMethodMayBeStatic
    def backward(self, tx, tcontext):
        """
        Inverse or a similar operation designed to act on outputs of the stage
        Defaults to no-op
        :param tx: A transformed value returned from the `forward` function
                    parts of it may have passed through other transforms
        :param tcontext: additional context needed for the backward step
        :return: An approximation of the original transformed object
        """
        return tx

    def roundtrip(self, x) -> ROut:
        out = self.forward(x)
        return ROut(self.backward(out.tx, out.tcontext), out.bitrate)


class FunctionalTransform(Transform):
    def __init__(self, forward, backward=None):
        super().__init__()
        self._forward = forward
        if backward is None:
            self._backward = lambda tx, tcontext: tx
        else:
            self._backward = backward

    def forward(self, x) -> FOut:
        return self._forward(x)

    def backward(self, tx, context):
        return self._backward(tx, context)


class SequenceTransform(FunctionalTransform):
    def __init__(self,
                 transforms: Sequence[Transform]):
        """
        :param transforms: these will run in sequence during forward
                           while their backward functions are executed in reverse order during backward

                           we assume that each forward returns a tuple (x, context)
        """

        def _forward(x) -> FOut:
            context = []
            bitrate = None
            for t in transforms:
                f_out = t.forward(x)
                context.append(f_out.tcontext)
                if f_out.bitrate is not None:
                    # in a sequence we trust the latest non-None bitrate
                    bitrate = f_out.bitrate
                x = f_out.tx
            return FOut(x, context, bitrate)

        def _backward(tx, tcontext):
            for t, c in reversed(list(zip(transforms, tcontext))):
                tx = t.backward(tx, c)
            return tx

        super().__init__(_forward, _backward)


class RepeatTransform(SequenceTransform):
    def __init__(self, n: int, transform: Transform):
        super().__init__([transform] * n)


class NoopTransform(Transform):
    def forward(self, x):
        return FOut(x)


noop = NoopTransform()