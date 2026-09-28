class EllipticCurve:
    def __init__(self, a, b, p): 
        # there should be contiditon check
        self.a, self.b, self.p = a, b, p

    def is_on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - (x**3 + self.a * x + self.b)) % self.p == 0

    def add(self, P, Q):
        p = self.p

        if P is None: # contiditon check
            return Q
        if Q is None: # contiditon check
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and (y1 + y2) % p == 0: # contiditon check
            return None
        if P == Q:
            m = (3 * x1 * x1 + self.a) * pow(2 * y1 % p, -1, p) % p
        else:
            m = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
        x3 = (m * m - x1 - x2) % p        # third intersection 
        y3 = (m * (x1 - x3) - y1) % p     # third points y
        return (x3, y3)

    def multiply(self, n, P):
        result = None      # start at O
        addend = P
        while n > 0:
            if n & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            n >>= 1
        return result

    def all_points(self):
        pts = [None]
        for x in range(self.p):
            for y in range(self.p):
                if self.is_on_curve((x, y)):
                    pts.append((x, y))
        return pts


def show(P):
    return "O" if P is None else str(P)


if __name__ == "__main__":
    curve = EllipticCurve(a=0, b=3, p=23)   # y^2 = x^3 + 3 (mod 23)

    P = (1, 2)
    Q = (7, 1)
    print("P      =", show(P), "on curve:", curve.is_on_curve(P))
    print("Q      =", show(Q), "on curve:", curve.is_on_curve(Q))
    print("P + Q  =", show(curve.add(P, Q)))
    print("P + P  =", show(curve.add(P, P)))
    print("P + -P =", show(curve.add(P, (1, 21)))) #manually calculated
    print("P + O  =", show(curve.add(P, None)))
    print()

    print("Number of points (including O):", len(curve.all_points()))
    print()

    print("Multiples of P until we get back to O:")
    n, R = 1, P
    while R is not None:
        print(f"  {n}P = {show(R)}")
        n += 1
        R = curve.add(R, P)
    print(f"  {n}P = O")
    print()

    print("Check double and add: 7P =", show(curve.multiply(7, P)))
