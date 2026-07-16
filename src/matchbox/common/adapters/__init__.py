"""Shared backend adapter contracts and machinery.

Code shared by every Matchbox backend: the cluster store protocol ABC,
and the SQL machinery that relational backends compose to implement it.

TODO: local client-side backends will depend on this package too.
"""
