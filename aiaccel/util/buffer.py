from __future__ import annotations

import copy
from typing import Any


class BufferData:
    def __init__(self, label: str) -> None:
        """
        Args:
            labal (str): A name of list.
        """
        self.label = label
        self.arr: list[Any] = []
        self._max_size = 65535

    def __call__(self, index: int) -> Any:
        return self.arr[index]

    def set_max_len(self, value: int) -> None:
        """Set any max size of this label length."""
        self._max_size = int(value)

    def add(self, value: Any) -> None:
        """Append any data."""
        if self.length == self._max_size:
            self.arr.pop(0)
        self.arr.append(value)

    @property
    def pre(self) -> Any:
        """Refers to the previous value."""
        if self.length >= 2:
            return self.arr[-2]
        else:
            return None

    @property
    def now(self) -> Any:
        """Refers to the current value."""
        return self.arr[-1]

    def clear(self) -> None:
        """Initialize list."""
        self.arr = []

    def replace(self, arr: list[Any]) -> None:
        """Replace to any list data."""
        self.clear()
        self.arr = copy.deepcopy(arr)

    @property
    def length(self) -> int:
        """Get list length."""
        return len(self.arr)

    @property
    def data(self) -> list[Any]:
        """Get the list data itself."""
        return self.arr

    def value(self, index: int) -> None:
        """Get any data in list.

        Args:
            index (int): A index of list.
        """
        return self.arr[index]

    def delete(self, index: int) -> None:
        """Delete any data in list.

        Args:
            index (int): A index of list.
        """
        self.arr.pop(index)

    @property
    def is_empty(self) -> bool:
        """Itself is empty or not"""
        if self.arr == []:
            return True
        else:
            return False

    def duplicate(self, value: Any) -> int:
        """Get index if exists duplicate data in list else -1.

        Args:
            value (Any): Check to see if the same value already exists.

        Returns:
            int: index value or -1.
        """
        for i in range(len(self.arr)):
            if self.arr[i] == value:
                return i
        return -1

    def delta(self) -> Any:
        """Get numerical difference."""
        return self.now - self.pre

    def point_diff(self, idx_pre: int, idx_now: int) -> Any:
        """Get the difference between any two points.

        Args:
            idx_pre (int): Any index value.
            idx_now (int): Any index value.

        Returns:
            any: difference between any two points.
        """
        return self.arr[idx_now] - self.arr[idx_pre]

    def has_difference(self, digit: int | None = None) -> bool:
        """Check there is a difference or not.

        Args:
            digit (int | None, optional): If this value is set, the value is
                rounded to the specified digit. Defaults to None.
        """
        if len(self.arr) >= 2:
            if digit is None:
                return self.pre != self.now
            else:
                return round(self.pre, digit) != round(self.now, digit)
        else:
            return False


class Buffer:
    """Buffer

    Args:
        labels (tuple) : Label names.

    Attributes:
        labels (list): A list of buffer data names.
        num_buff (int): A length of labels.
        d (dict): A dictionary for accessing arbitrary buffer data

    Example:
     ::

        # create buffer
        buff = Buffer(["data1", "data2", "data3"])

        # add data
        buff.d["data1"].add(x)
        buff.d["data2"].add(x)
        buff.d["data3"].add(x)
        # or
        buff.add("data1", x)
        buff.add("data2", x)
        buff.add("data3", x)
    """

    def __init__(self, *labels: Any) -> None:
        self.labels = labels[0]
        self.num_buff = len(self.labels)
        self.d = {}
        for i in range(self.num_buff):
            self.d[self.labels[i]] = BufferData(self.labels[i])

    def add(self, label: str, value: Any) -> None:
        """add a data to any buffer.

        Args:
            label (str): A target buffer labele.
            value (any): additional.
        """
        self.d[label].add(value)

    def delete(self, label: str, index: int) -> None:
        """Delete a any data in any buffer.

        Args:
            label (str): A target label.
            index (int): A Index to be deleted in the target buffer.
        """
        self.d[label].delete(index)

    def clear(self, label: str) -> None:
        """Delete all buffer data of the target.

        Args:
            label (str): A target label.
        """
        self.d[label].clear()
