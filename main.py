from engine import Value

def main():
    print("Hello from micrograd-vibecode!")
    
    # Verification of operations
    a = Value(2.0)
    b = Value(3.0)
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    c = a + b
    print(f"a + b = {c}")
    
    d = a * b
    print(f"a * b = {d}")
    
    e = a ** 3
    print(f"a ** 3 = {e}")
    
    f = a / b
    print(f"a / b = {f}")
    
    g = a - b
    print(f"a - b = {g}")
    
    # Complex expression
    h = (a * b + c) / a
    print(f"(a * b + c) / a = {h}")


if __name__ == "__main__":
    main()
