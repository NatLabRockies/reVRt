# Path optimization with custom cost function


## Inbox (ideas to be organized)

Ideally this section should be empty, so whenever there is a chance, this
points should be organized and moved to the appropriate section.

- Miscellaneous topics to dive deeper later
  - profile.release
    - incremental
    - lto
      - lto off when running dhat
  - Profile with: export RUSTFLAGS="-C target-cpu=native"
  - Supported names: rustc --print target-cpus
  - Profile with `Cargo.toml`:
    ```
    [profile.release]
    debug = "line-tables-only"
    ```

- Profiling memory usage:
  - Using dhat:
    - cargo run --release -p revrt-cli --features dhat-heap -- -vv -d ../transmission_costs.zarr --cost-function='{"cost_layers": [{"layer_name": "fmv_dollar_per_acre"}, {"layer_name": "swca_natural_resources_risk_2"}]}' --start 20012,40000 --end 20012,40100 --cache-size=250000
      Which gives this:
      ```
      dhat: Total:     730,183,928 bytes in 5,192,541 blocks
      dhat: At t-gmax: 80,245,023 bytes in 428 blocks
      dhat: At t-end:  158,992 bytes in 268 blocks
      dhat: The data has been saved to dhat-heap.json, and is viewable with dhat/dh_view.html
      ```
  - Somes results:
    - distance: t-gmax (Total)
    - 0: 2.7MB
    - 1: 56MB
    - 2: 56MB
    - 10: 56MB (120MB
    - 20 (1 feature): 32MB
    - 20 (5 features, 1 chunks): 104MB
    - 20 (5 features, 2 chunks): 104MB
    - 20 (5 features, 4 chunks): 104MB
    - 25: 56MB (150MB)
    - 100: 56MB (496MB)
    - 200: 56MB (1.8 GB)
    - 300 (5 features, 1 chunk): 104MB (3.6 GB), 442s
    - 500 (5 features, 3 chunks): 161MB (9.7GB), 1193s
    - 1000 (5 features, 6 chunks): 230MB (34GB), 4246s
    - 1500 (5 features, 8 chunks): 794MB (96GB), 12144s - Cache size 125MB
- Benchmarking with smaply
  - cargo install --locked samply
  - cargo build --release -p revrt-cli
  - samply record  ./target/release/revrt-cli  -vv -d ../transmission_costs.zarr --cost-function='{"cost_layers": [{"layer_name": "fmv_dollar_per_acre"}, {"layer_name": "swca_natural_resources_risk_2"}]}' --start 20500,40500 --end 20500,40600 --cache-size=250000
- Benchmarking with criterion:
  - Instructions to run benchmark locally:
    - `cargo bench --bench standard`
    - `cargo bench --bench serial -- --save-baseline mybaseline`
    - `cargo criterion --output-format bencher --benches`
  - Visualize output at `target/criterion/`
- Use a cache for the chunks.
- Does it make sense a cache on the final weight calculated? Maybe
  a HashMap with (x, y), and rolling the oldest out to minimize
  the overhead by tracking use such as LRU?
- Should it try to read ahead? Maybe a trigger when get too close
  to the edge of a chunk, so it already request the next neighbor?
- Certainly async!
- Read the chunk is probably the most expensive operation, and
  calculate the cost for each grid point is probably next. Thus:
  - Request one point in the database;
  - From the point, designate which chunk it is;
  - Load the full chunk (I don't think it is possible or efficient
    don't do a full chunk at once), for all required variables;
  - Calculate the cost for all points in the chunk in parallel.
    This is a place easy to parallelize. The question is how the
    pathfinding algorithm prioritize the next target. Due to I/O,
    depth first search can't be a good deal because read a full
    chunk to use only a line of points.
  - Some implementations require Copy, thus float is not possible.
  - Goal can be multiple targets. Return the best solution for each target
    and return the best of the best. Maybe return in order of best cost.

## Contract of service:

This project is commited to deliver:
- Find the best path. Local or partial optimal is not enough.
- Able to scan routes on national scale with resolution O[10m].
- Be at least as fast as the former application.
- While taking the best from HPC, be able to run in a regular laptop.
- Able to scan multiple solutions concurrently.
- Be able to use it as an analysis tool without requiring advanced knowledge on Python or Rust.
- Provide a smooth transition such that users from former application should require at most 1hr to adapt to this new solution.

## Priorities:

1. Correctness. Give a correct answer at any cost.
   Simplifications:
   - Don't worry about memory requirements or speed.
   - All serial and synchronous.
   - Tests are welcome, but not required.
   - unwrap() is fine.
   - Ignore the first pixel in the boundary (i.e. i=0 & i=len-1).
   - Use as many public resources as convenient. Avoid custom solutions
     at this stage.
2. Python interface.
   Allow realworld usage of the code even if it is not the most efficient.
   The goal here is to minimize the effort on maintaining the Python
   library to allow diverting the effort to the Rust code.
3. Asynchronous operations.
   - At least the I/O.
   - Depending on the final design, the cost calculation might be a good
     case for that as well.
   - It is annoying to have that before implementing the tests, but it
     might impose too many changes in the code and probably would result
     in re-writing the tests.
4. Harness the code with minimum tests.
   - Focus on high level results, since the low level specific operations
     might change soon.
   - Some basic integration tests to build the grounds for future. Keep
     in mind that this might get heavy.
   - Initiate the samples features so as we evolve the code, we can keep
     track on the edge cases right the way. The goal here is to add and
     modify the samples right the way as a new edge case is identified.
5. Manageable memory footprint.
   We don't need to limit ourselves to the smallest memory footprint
   possible, but we should keep in mind that the memory requirements.
   It's OK to use some memory, but it should be possible to run it with
   a reasonable ammount.
   - What is a reasonable ammount? 30GB? It should not necessarily require
     HPC, but feasible in a regular computer, even if slow.
   - Be able to run in a laptop must not compromise the performance in
     the HPC. If a choice is to be made, the priority is to take full
     advantage of extra resouces, and pay an extra price in the regular
     laptop. I.e. if has access to 150GB or 100 cores, take full advantage
     of that, while keeping an alternative, even if costly, for a laptop.
   - We don't need to be ready for infinite memory requirements. That
     would probably create restrictions along the design. What is a fair
     limit? Expect 1TB of input data, i.e. all variables in multiple
     datasets required to estimate the cost and path?
     (20 variables f64, with 50m resolution)
6. Speed.
   - Assume availability of multiple cores, so parallelism might
     be an advantage.
   - Given the grid size, I/O is probably the bottleneck.
