
from engine import Value
import math

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.data + z.data * x.data + x.data**2
    y = q / x.data
    
    # expected result using simple python floats
    x_val = -4.0
    z_val = 2 * x_val + 2 + x_val
    q_val = z_val + z_val * x_val + x_val**2
    y_val = q_val / x_val
    
    assert y == y_val # Comparing Value with float should fail if I don't handle __eq__, but comparing data is safer.
    # Actually wait, I haven't implemented __eq__ or comparison with primitives in a smart way yet.
    # The prompt asked for specific ops.
    # Let's write tests that check .data explicitly to be safe and robust.

def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0

def test_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0

def test_pow():
    a = Value(2.0)
    c = a ** 3
    assert c.data == 8.0

def test_div():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    assert c.data == 2.0

def test_sub():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0

def test_neg():
    a = Value(5.0)
    b = -a
    assert b.data == -5.0

def test_complex_expression():
    # (a * b + c) / a
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0) # slightly different from main.py, making it clean
    
    # (2 * 3 + 1) / 2 = 7 / 2 = 3.5
    res = (a * b + c) / a
    assert res.data == 3.5

def test_relu():
    a = Value(-1.0)
    assert a.relu().data == 0.0
    b = Value(1.0)
    assert b.relu().data == 1.0

def test_graph_structure():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    
    assert a.grad == 0.0
    assert b.grad == 0.0
    assert c.grad == 0.0
    
    assert a in c._prev
    assert b in c._prev
    assert c._op == "*"


