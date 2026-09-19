"""Pure numerical core for the stochastic engine (see
/stochastic-engine-proposal.md). Every module here works over plain numpy
arrays and JSON-shaped dicts -- never pl.DataFrame, never a custom packet
class -- the same convention blocks/modelling.py uses for its "model" port
(a plain dict, not a pickled estimator): a distribution/proxy/dependency
object stays inspectable over the API's generic /value endpoint and never
needs a special deserializer downstream.

`modelmaker/blocks/stochastic.py` is the thin BlockSpec layer on top: it
does the pl.DataFrame <-> numpy conversion at the edges and calls into
these modules for the actual math.
"""
