from __future__ import annotations

import string
from typing import Any

import numpy as np

coef: dict = {"r": 1.0, "ic": -0.5, "oc": 0.5, "e": 2.0, "s": 0.5}
name_rng = np.random.RandomState()


class Vertex:
    def __init__(self, xs: np.ndarray, value=None):
        self.xs = xs
        self.value = value
        self.id = self.generate_random_name()

    def generate_random_name(self, length: int = 10) -> str:
        if length < 1:
            raise ValueError("Name length should be greater than 0.")
        rands = [name_rng.choice(list(string.ascii_letters + string.digits))[0] for _ in range(length)]
        return "".join(rands)

    @property
    def coordinates(self) -> np.ndarray:
        return self.xs

    def set_value(self, value: Any) -> None:
        self.value = value

    def set_id(self, id: str):
        self.id = id

    def set_new_id(self):
        self.id = self.generate_random_name()

    def set_xs(self, xs: np.ndarray):
        self.xs = xs

    def update(self, xs: np.ndarray, value) -> None:
        self.xs = xs
        self.value = value

    def __add__(self, other):  # Add +
        if isinstance(other, Vertex):
            new_vertex = Vertex(self.coordinates + other.coordinates)
            return new_vertex
        try:
            new_vertex = Vertex(self.coordinates + other)
            return new_vertex
        except TypeError:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):  # Subtract -
        if isinstance(other, Vertex):
            new_vertex = Vertex(self.coordinates - other.coordinates)
            return new_vertex
        try:
            new_vertex = Vertex(self.xs - other)
            return new_vertex
        except TypeError:
            raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):  # Multiply *
        new_vertex = Vertex(self.xs * other)
        new_vertex.set_id(self.id)
        return new_vertex

    def __eq__(self, other):  # Equal ==
        if isinstance(other, Vertex):
            return self.value == other.value
        try:
            return self.value == other
        except TypeError:
            raise TypeError("Unsupported operand type for ==")

    def __ne__(self, other):  # Not Equal !=
        if isinstance(other, Vertex):
            return self.value != other.value
        try:
            return self.value != other
        except TypeError:
            raise TypeError("Unsupported operand type for !=")

    def __lt__(self, other):  # Less Than <
        if isinstance(other, Vertex):
            return self.value < other.value
        try:
            return self.value < other
        except TypeError:
            raise TypeError("Unsupported operand type for <")

    def __le__(self, other):  # Less Than or Equal <=
        if isinstance(other, Vertex):
            return self.value <= other.value
        try:
            return self.value <= other
        except TypeError:
            raise TypeError("Unsupported operand type for <=")

    def __gt__(self, other):  # Greater Than >
        if isinstance(other, Vertex):
            return self.value > other.value
        try:
            return self.value > other
        except TypeError:
            raise TypeError("Unsupported operand type for >")

    def __ge__(self, other):  # Greater Than or Equal >=
        if isinstance(other, Vertex):
            return self.value >= other.value
        try:
            return self.value >= other
        except TypeError:
            raise TypeError("Unsupported operand type for >=")


class Simplex:
    def __init__(self, simplex_coordinates: np.ndarray):
        self.n_dim = simplex_coordinates.shape[1]
        self.vertices: list[Vertex] = []
        self.centroid: Vertex = None
        self.coef = coef
        for xs in simplex_coordinates:
            self.vertices.append(Vertex(xs))

    def get_simplex_coordinates(self) -> np.ndarray:
        return np.array([v.xs for v in self.vertices])

    def set_value(self, vertex_id: str, value: Any) -> bool:
        for v in self.vertices:
            if v.id == vertex_id:
                v.set_value(value)
                return True
        return False

    def order_by(self):
        order = np.argsort([v.value for v in self.vertices])
        self.vertices = [self.vertices[i] for i in order]

    def calc_centroid(self):
        self.order_by()
        xs = self.get_simplex_coordinates()
        self.centroid = Vertex(xs[:-1].mean(axis=0))

    def reflect(self) -> Vertex:
        xr = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["r"])
        return xr

    def expand(self) -> Vertex:
        xe = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["e"])
        return xe

    def inside_contract(self) -> Vertex:
        xic = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["ic"])
        return xic

    def outside_contract(self) -> Vertex:
        xoc = self.centroid + ((self.centroid - self.vertices[-1]) * self.coef["oc"])
        return xoc

    def shrink(self) -> list[Vertex]:
        for i in range(1, len(self.vertices)):
            self.vertices[i] = self.vertices[0] + (self.vertices[i] - self.vertices[0]) * self.coef["s"]
        return self.vertices


class Value:
    def __init__(self, id: str | None = None, value: Any = None):
        self.id: str = id
        self.value: Any = value


class Store:
    def __init__(self):
        self.r: Vertex | None = None  # reflect
        self.e: Vertex | None = None  # expand
        self.ic: Vertex | None = None  # inside_contract
        self.oc: Vertex | None = None  # outside_contract
        self.s: list[Vertex] | None = None  # shrink


class NelderMead:
    def __init__(self, initial_parameters: Any = None):
        self.simplex: Simplex = Simplex(initial_parameters)
        self.state = "initialize"
        self.store = Store()
        self.waits = {
            "initialize": self.simplex.n_dim + 1,
            "shrink": self.simplex.n_dim + 1,
            "expand": 1,
            "inside_contract": 1,
            "outside_contract": 1,
            "reflect": 1,
        }
        self.n_waits = self.waits[self.state]

    def get_n_waits(self) -> int:
        return self.n_waits

    def get_n_dim(self) -> int:
        return self.simplex.n_dim

    def get_state(self) -> str:
        return self.state

    def change_state(self, state: str) -> None:
        self.state = state

    def set_value(self, vertex_id: str, value: float | int):
        self.simplex.set_value(vertex_id, value)

    def initialize(self) -> list(Value):
        self.n_waits = self.waits["initialize"]
        return self.simplex.vertices

    def after_initialize(self, yis: list[Value]):
        for y in yis:
            self.simplex.set_value(y.id, y.value)
        self.change_state("reflect")

    def reflect(self) -> Vertex:
        self.n_waits = self.waits["reflect"]
        self.simplex.calc_centroid()
        self.store.r = self.simplex.reflect()
        return self.store.r

    def after_reflect(self, yr: Value):
        self.store.r.set_value(yr.value)
        if self.simplex.vertices[0] <= self.store.r < self.simplex.vertices[-2]:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
            self.change_state("reflect")
        elif self.store.r < self.simplex.vertices[0]:
            self.change_state("expand")
        elif self.simplex.vertices[-2] <= self.store.r < self.simplex.vertices[-1]:
            self.change_state("outside_contract")
        elif self.simplex.vertices[-1] <= self.store.r:
            self.change_state("inside_contract")
        else:
            self.change_state("reflect")

    def expand(self) -> Vertex:
        self.n_waits = self.waits["expand"]
        self.store.e = self.simplex.expand()
        return self.store.e

    def after_expand(self, ye: Value):
        self.store.e.set_value(ye.value)
        if self.store.e < self.store.r:
            self.simplex.vertices[-1].update(self.store.e.coordinates, self.store.e.value)
        else:
            self.simplex.vertices[-1].update(self.store.r.coordinates, self.store.r.value)
        self.change_state("reflect")

    def inside_contract(self) -> Vertex:
        self.n_waits = self.waits["inside_contract"]
        self.store.ic = self.simplex.inside_contract()
        return self.store.ic

    def after_inside_contract(self, yic: Value):
        self.store.ic.set_value(yic.value)
        if self.store.ic < self.simplex.vertices[-1]:
            self.simplex.vertices[-1].update(self.store.ic.coordinates, self.store.ic.value)
            self.change_state("reflect")
        else:
            self.change_state("shrink")

    def outside_contract(self) -> Vertex:
        self.n_waits = self.waits["outside_contract"]
        self.store.oc = self.simplex.outside_contract()
        return self.store.oc

    def after_outside_contract(self, yoc: Value):
        self.store.oc.set_value(yoc.value)
        if self.store.oc <= self.store.r:
            self.simplex.vertices[-1].update(self.store.oc.coordinates, self.store.oc.value)
            self.change_state("reflect")
        else:
            self.change_state("shrink")

    def shrink(self) -> list[Vertex]:
        self.n_waits = self.waits["shrink"]
        self.store.s = self.simplex.shrink()
        return self.store.s

    def aftter_shrink(self, yss: list[Value]):
        for ys in yss:
            if not self.simplex.set_value(ys.id, ys.value):
                raise "Error: vertex is not found."
        self.change_state("reflect")

    def search(self) -> list[Vertex]:
        if self.state == "initialize":
            xs = self.initialize()
            # print(f"initialize: {[v.coordinates for v in xs]}")
            self.change_state("initialize_pending")
            return xs

        elif self.state == "initialize_pending":
            return []

        elif self.state == "reflect":
            x = self.reflect()
            self.change_state("reflect_pending")
            # print(f"reflect: {x.coordinates}")
            return [x]

        elif self.state == "reflect_pending":
            return []

        elif self.state == "expand":
            x = self.expand()
            self.change_state("expand_pending")
            # print(f"expand: {x.coordinates}")
            return [x]

        elif self.state == "expand_pending":
            return []

        elif self.state == "inside_contract":
            x = self.inside_contract()
            self.change_state("inside_contract_pending")
            # print(f"inside_contract: {x.coordinates}")
            return [x]

        elif self.state == "inside_contract_pending":
            return []

        elif self.state == "outside_contract":
            x = self.outside_contract()
            self.change_state("outside_contract_pending")
            # print(f"outside_contract: {x.coordinates}")
            return [x]

        elif self.state == "outside_contract_pending":
            return []

        elif self.state == "shrink":
            xs = self.shrink()
            self.change_state("shrink_pending")
            # print(f"shrink: {[v.coordinates for v in xs]}")
            return xs

        elif self.state == "shrink_pending":
            return []
