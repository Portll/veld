Storage abstraction entry point for backend-agnostic persistence.

This folder is the intended seam between recall/graph logic and concrete
 backends. The initial step defines traits and capability metadata only; live
 runtime wiring still goes through the existing RocksDB compatibility path.