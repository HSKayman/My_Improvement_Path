# Purpose: Project Euler problem 29: Distinct powers.
L = 100
r = range(2, L+1)

print("{}".format(len({a**b for a in r for b in r})))