from aiaccel.util import Buffer


def test_set_max_len():
    buff = Buffer(["test"])

    assert buff.d["test"]._max_size == 65535
    buff.d["test"].set_max_len(255)
    assert buff.d["test"]._max_size == 255


def test_add():
    buff = Buffer(["test"])
    print(buff.d["test"])
    assert buff.d["test"].data == []
    buff.add("test", 1)
    assert buff.d["test"].data[0] == 1
    assert buff.d["test"](0) == 1


def test_add_lengthover():
    buff = Buffer(["test"])
    buff.d["test"].set_max_len(2)
    assert buff.d["test"]._max_size == 2

    buff.add("test", 1)
    buff.add("test", 2)
    assert buff.d["test"].length == 2
    assert buff.d["test"](0) == 1
    assert buff.d["test"](1) == 2

    buff.add("test", 3)
    assert buff.d["test"].length == 2
    assert buff.d["test"](0) == 2
    assert buff.d["test"](1) == 3

    buff.add("test", 4)
    assert buff.d["test"].length == 2
    assert buff.d["test"](0) == 3
    assert buff.d["test"](1) == 4


def test_pre():
    buff = Buffer(["test"])
    buff.add("test", 6)
    assert buff.d["test"].pre is None
    buff.add("test", 7)
    assert buff.d["test"].pre == 6


def test_now():
    buff = Buffer(["test"])
    buff.add("test", 4)
    assert buff.d["test"].now == 4


def test_delete():
    buff = Buffer(["test"])
    buff.add("test", 1)
    assert buff.d["test"](0) == 1
    assert buff.d["test"].length == 1
    buff.delete("test", 0)
    assert buff.d["test"].length == 0


def test_clear():
    buff = Buffer(["test"])
    buff.add("test", 1)
    assert buff.d["test"](0) == 1
    assert buff.d["test"].length == 1
    buff.clear("test")
    assert buff.d["test"].length == 0


def test_replace():
    buff = Buffer(["test"])
    buff.add("test", 1)
    buff.add("test", 2)
    buff.add("test", 3)
    buff.add("test", 4)
    buff.add("test", 5)
    assert buff.d["test"].data == [1, 2, 3, 4, 5]
    new_arr = [6, 7, 8, 8, 9]
    buff.d["test"].replace(new_arr)
    assert buff.d["test"].data == [6, 7, 8, 8, 9]


def test_value():
    buff = Buffer(["test"])
    buff.add("test", 1)
    buff.add("test", 2)
    assert buff.d["test"].value(0) == 1
    assert buff.d["test"].value(1) == 2


def test_is_empty():
    buff = Buffer(["test"])
    assert buff.d["test"].is_empty is True
    buff.add("test", 1)
    assert buff.d["test"].is_empty is False


def test_duplicate():
    buff = Buffer(["test"])
    buff.add("test", 1)
    buff.add("test", 2)
    buff.add("test", 3)
    assert buff.d["test"].duplicate(1) == 0
    assert buff.d["test"].duplicate(2) == 1
    assert buff.d["test"].duplicate(3) == 2
    assert buff.d["test"].duplicate(4) == -1


def test_delta():
    buff = Buffer(["test"])
    buff.add("test", 1)
    buff.add("test", 5)
    assert buff.d["test"].delta() == 4


def test_point_diff():
    buff = Buffer(["test"])
    buff.add("test", 1)
    buff.add("test", 5)
    assert buff.d["test"].point_diff(0, 1) == 4


def test_has_difference():
    buff = Buffer(["test"])
    assert buff.d["test"].has_difference() is False

    buff.add("test", 1.12)
    buff.add("test", 5.45)
    assert buff.d["test"].has_difference() is True
    assert buff.d["test"].has_difference(digit=1) is True

    buff.add("test", 1.12)
    buff.add("test", 1.12)
    assert buff.d["test"].has_difference() is False
    assert buff.d["test"].has_difference(digit=1) is False

    buff.add("test", 1.12)
    buff.add("test", 1.13)
    assert buff.d["test"].has_difference() is True
    assert buff.d["test"].has_difference(digit=1) is False
