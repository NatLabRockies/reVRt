# Routing Layer Concept Model

``reVRt`` is designed to help you compute an "optimal" route from
point ``A`` to ``B``. To define what we mean by ``optimal``, we
use high-resolution geospatial layers with spatially-distinct
"cost" values. The "cost" values can represent actual dollar costs
or any other quantity that you may wish to study.

The optimal route is always computed through a single layer, but
this layer may be composed of several customizable components:
**Cost Layers**, **Friction layers**, and **Barrier Layers**.
In the following sections, we break down each of these components
and how you can use them to build out the scenario that you are
interested in studying.

## Cost Layers

Cost Layers are the core routing layer used by ``reVRt``. The
aggregate of these layers is the main driver for the optimal
route solution returned by the code. These layers should represent
the total cost (dollar or otherwise) to cross a single pixel
edge-to-edge (**not** diagonally). ``reVRt`` internally adjusts
these cost values if the path goes diagonally through the pixel.


### Building cost layers
A single cost layer could be defined as follows:

```json5
{
    "layer_name": "my_cost_layer",
    "multiplier_layer": "my_bool_layer",
    "multiplier_scalar": 1.04  // adjust for inflation
}
```

This would create a cost layer from ``"my_cost_layer"``, which would
have a mask layer applied to it (``"my_bool_layer"``) along with a
scalar that would adjust the costs (for example, a value of ``1.04`` can
account for inflation). There are several more options you can include
for a single layer; they are all documented
[here](https://natlabrockies.github.io/reVRt/_cli/reVRt.html#revrt-route-points:~:text=cost_layerslist).

For maximum flexibility, you can specify multiple such layers that all
aggregate to form the final routing cost layer:

```json5
"cost_layers": [
    {
        "layer_name": "my_cost_layer",
        "multiplier_layer": "my_bool_layer",
        "multiplier_scalar": 1.04  // adjust for inflation
    },
    {
        "layer_name": "local_costs_mask",
        "multiplier_scalar": 10000  // cost (per pixel) to route through a local town
    },
    // ...
]
```

Each cost layer is built independently, and they are all summed at the
end to create the cost routing layer. Output routes strive to minimize
the total sum of the values in this layer along the output route.

### Invalid costs
One of the restrictions of the routing algorithm is that all costs must
be strictly positive. Negative, zero, or NaN costs are not allowed (we
call them "invalid" costs).

In an ideal world, all the input cost data would have values {math}`\gt 0` to
satisfy this requirement. Unfortunately, real-world data is rarely
comprehensive enough to have a valid cost estimate across an entire domain.
This problem is particularly amplified when working with high-resolution data
across a large extent like CONUS.

To reduce headaches for users, ``reVRt`` has two options to handle cost values
{math}`\leq 0`. The first (and default) option is to ignore these cells entirely.
This means routes cannot pass through them (they act as a quasi-barrier, but
within the cost layer itself). This is likely the behavior that the vast majority
of users will want, since these "invalid" costs often occur at the edge of the
domain anyways.

Sometimes, however, the presence of invalid costs can prevent a route from being
completed (imagine a start point completely surrounded by invalid costs, either
locally or further into the domain). In these rare cases, it might be useful to
allow routes to be formed over the invalid costs in order to get any result back.
In this cases, users may specify ``ignore_invalid_costs: false`` in their
configuration, and ``reVRt`` will allow routes to pass over these invalid cost
pixels, minimizing the route's exposure to them.


## Friction Layers
Friction layer behave similarly to cost layers, except that **their
contribution is not included in the output costs**. Therefore you can
use friction layers to influence the "shape" of the final route without
including any the friction values in the output cost.

### Contribution to Routing
Friction is added to the cost routing layer using the following equation:

```{math}
R = C * (1 + F)
```

where {math}`R` is the final routing layer, {math}`C` is the cost layer built
in the previous section, and {math}`F` is the built-out friction layer (see
below for details on building a friction layer).

Based on this formulation, we can see that positive friction values can
be used to strongly discentivise routes through certain pixels. On the
other hand, friction values near ``-1`` can be used to represent incentives
(e.g. routing along an existing right-of-way). ``reVRt`` ensures that
individual friction values are {math}`\gt -1` (i.e. {math}`(1 + F) > 0` is
upheld), so the cost values themselves can never flip signs.

### Building friction layers
Friction layers are built similarly to cost layers. A single friction
layer could be defined as follows:

```json5
{
    "multiplier_layer": "my_friction_region",
    "multiplier_scalar": 10  // moderately avoid
}
```

One major difference is that there is no ``"layer_name"`` input, since
the friction layer itself is being multiplied onto the final cost routing
layer (see the section above). As before, there are several more options
you can include for a single friction layer; they are all documented
[here](https://natlabrockies.github.io/reVRt/_cli/reVRt.html#revrt-route-points:~:text=friction_layerslist).

For maximum flexibility, you can specify multiple friction layers that all
aggregate to form the final friction layer:

```json5
"friction_layers": [
    {
        "multiplier_layer": "my_friction_region",
        "multiplier_scalar": 10  // moderately avoid
    },
    {
        "multiplier_layer": "really_avoid_this",
        "multiplier_scalar": 1000  // strongly discentivise
    },
    {
        "multiplier_layer": "highway_mask",
        "multiplier_scalar": -0.5  // encourage routing here
    },
    // ...
]
```
As mentioned before, the friction multiplier can be negative values in
order to incentivize routing along certain pixels. You can specify multiple
such layers. ``reVRt`` processes each layer individually and then aggregates
everything to determine the final friction layer. After the aggregation, all
friction values are clamped to be {math}`\gt -1`, such that friction can
never create invalid cost values in the cost layer.

## Barrier Layers
Barrier layers represent pixels that routes must not cross. Unlike frictions,
which can be crossed with a great penalty, barrier pixels will block a route
completely (unless you explicitly disable this using ranked barrier layers,
as described below).

### Building barrier layers
Barrier layers are geospatial layers just like costs or frictions that are
paired with a definition of what values should act as the barrier (the standard
comparison operators are allowed: ``>``, ``>=``, ``<``, ``<=``, ``==``). Any
pixel with a value that satisfies the comparison operator will be treated as a
barrier that cannot be crossed by a route. For example, this configuration:

```json5
{"layer_name": "slope", "barrier_values": ">=15"}
```

would tell the routing algorithm that any pixels with a value {math}`\ge 15`
in the ``slope`` layer should be completely avoided. As with all the other
layers, you can specify multiple barriers to be considered during routing:

```json5
"barrier_layers": [
    {"layer_name": "slope", "barrier_values": ">=15"},
    {"layer_name": "barrier_bool_mask", "barrier_values": "==1"},
    // ...
]
```

You can also repeat barrier layer entires with different barrier values; this
becomes useful when applying soft barriers.

### Soft Barriers
If a route you are interested is impossible to create because of the barrier
layers that you have specified, ``reVRt`` will return an empty result (no route).
Sometimes it's desirable to relax one or more barrier layer assumptions in order
to get some sort of route result.

To do this, ``reVRt`` allows you to rank barriers by importance. When a route
can not be determined, ``reVRt`` will drop the lowest-ranking barrier layer and
attempt to re-compute the route. This process is repeated until a route is found
or all barrier layers with a rank have been dropped.

To specify that a layer can be dropped in order to compute a route (i.e. a soft
barrier), you have to provide a ``barrier_importance`` ranking, like so:

```json5
"barrier_layers": [
    {"layer_name": "slope", "barrier_values": ">=15", "barrier_importance": 10},
    {"layer_name": "barrier_bool_mask", "barrier_values": "==1", "barrier_importance": 1},
    // ...
]
```

With the configuration above, ``reVRt`` will first attempt to compute a route by
applying both ``slope`` and ``barrier_bool_mask`` barriers. If a route cannot be
determined, the ``barrier_bool_mask`` layer will be dropped, since it ranks lower
than the ``slope`` layer. If a route is still not found with only the ``slope``
barrier, it will be dropped as well.

You can also specify that a barrier should never be dropped by leaving out the
``barrier_importance`` key altogether. This can be combined with soft barriers:

```json5
"barrier_layers": [
    {"layer_name": "slope", "barrier_values": ">=15", "barrier_importance": 10},
    {"layer_name": "barrier_bool_mask", "barrier_values": "==1", "barrier_importance": 1},
    {"layer_name": "important_barrier", "barrier_values": "<0.5"},
    // ...
]
```

With this configuration, ``reVRt`` will drop the ``slope`` and ``barrier_bool_mask``
layers to try to find a route, but will always keep the ``important_barrier`` layer
as a barrier.
