from pathlib import Path

path = Path("docs/models/gwr.md")
text = path.read_text(encoding="utf-8")
old = """After the bandwidth is known, standard GWR calibration, local R² calculation and target-location prediction evaluate target-to-training distances in bounded row blocks rather than retaining a complete pairwise matrix. With a **numeric bandwidth** and `compute_hat_matrix=False`, the calibration distance working set therefore scales linearly with the number of training observations instead of storing another `n × n` array. Automatic bandwidth selection remains a separate exception: the current CV/AIC/AICc/BIC selectors precompute the training pairwise distance matrix so they can score many candidate bandwidths without recomputing distances. If automatic selection is the memory bottleneck, use a defensible numeric bandwidth or `ScalableGWR` until the selector path is also streamed.
"""
new = """Standard GWR calibration, local R² calculation, target-location prediction, and automatic CV/AIC/AICc/BIC bandwidth selection evaluate distances in bounded row blocks rather than retaining a complete pairwise distance matrix. With `compute_hat_matrix=False`, distance working memory therefore scales linearly with the number of training observations for a fixed block size instead of storing another `n × n` array. Automatic bandwidth selection deliberately recomputes bounded distance blocks while scoring candidate bandwidths, trading additional distance-calculation time for bounded memory; adaptive automatic search-range construction does not require pairwise distances, while fixed-distance automatic bounds are obtained from a streamed minimum/maximum positive-distance scan.
"""
assert text.count(old) == 1
path.write_text(text.replace(old, new), encoding="utf-8")
