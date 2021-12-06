from ..util import is_numeric_literal
import pdb


class Interval:
    def __init__(self, low=0, upper=0):
        self.low = low
        self.upper = upper

    def __str__(self) -> str:
        return f'[{self.low}, {self.upper}]'
    __repr__ = __str__

    def __add__(self, other):
        if is_numeric_literal(other):
            return self + Interval(other, other)

        return Interval(self.low + other.low, self.upper + other.upper)

    def __radd__(self, other):
        if is_numeric_literal(other):
            return Interval(other, other) + self

        return Interval(self.low + other.low, self.upper + other.upper)

    def __mul__(self, other):
        if is_numeric_literal(other):
            return self * Interval(other, other)
        return Interval(
            low=min(self.low * other.low, self.low * other.upper,
                    self.upper * other.low, self.upper * other.upper),
            upper=max(self.low * other.low, self.low * other.upper,
                      self.upper * other.low, self.upper * other.upper),
        )

    def __rmul__(self, other):
        if is_numeric_literal(other):
            return Interval(other, other) * self
        return Interval(
            low=min(self.low * other.low, self.low * other.upper,
                    self.upper * other.low, self.upper * other.upper),
            upper=max(self.low * other.low, self.low * other.upper,
                      self.upper * other.low, self.upper * other.upper),
        )

    def __sub__(self, other):
        if is_numeric_literal(other):
            return self - Interval(other, other)
        return Interval(low=self.low - other.upper, upper=self.upper - other.low)

    def relu(self):
        return Interval(low=max(0, self.low), upper=max(0, self.upper))

    def intersect(self, other):
        return Interval(max(self.low, other.low), min(self.upper, other.upper))

    def union(self, other):
        return Interval(min(self.low, other.low), max(self.upper, other.upper))

    def __eq__(self, other):
        return self.low == other.low and self.upper == other.upper
