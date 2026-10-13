# Scientific scope

MapSy generates local-environment descriptor maps and supports exploratory dimensionality reduction and clustering. Cluster counts and assignments require validation for the intended scientific application.

The classical and surface-referenced tutorials have different probe models. Compare atomic geometry, probe positions, descriptor radius/order and species vocabulary before interpreting differences. For Pt111, known anchors are atop, bridge, fcc and hcp. Their resolution depends on layers captured and cutoff; increasing radius also requires a sufficiently large periodic repeat. The present descriptor uses one minimum image per atom.

Silhouette and Davies–Bouldin curves document the explored geometric partitions. The experimental study evaluates missing descriptor information and known-site consistency separately; it does not select a universal physical k. Protein compositional codes are inferred species weights, distinct from the separately computed physicochemical fields. Bundled proteins omit calcium, water and hydrogens; the q/r field is not a complete force-field assignment.

These examples support descriptor exploration, visualization and downstream modeling. They do not require energy calculations. Future work on rare-environment-aware coverage, symmetry constraints and hierarchical connectivity belongs to the experimental section. No new clustering method is asserted as validated in this release.
