---
hide:
  - navigation
  - toc
---

# pyGWRx { .pygx-home-title }

<div class="pygx-home">

<section class="pygx-hero">
  <div class="pygx-hero__inner">
    <h2>
      Geographically<br>
      weighted modelling,<br>
      <span>refined for research.</span>
    </h2>
    <p class="pygx-hero__lead">
      pyGWRx is a Python toolkit for spatially varying relationships—<br>
      multiscale, robust, and extensible.
    </p>
    <div class="pygx-hero__actions">
      <a class="md-button md-button--primary" href="getting-started/quickstart/">Getting started</a>
      <a class="md-button" href="getting-started/">User Guide</a>
    </div>
  </div>
</section>

<section class="pygx-section">
  <span class="pygx-section__kicker">Choose your path</span>
  <h2>Move from question to result</h2>
  <p class="pygx-section__intro">
    The documentation is organized around the task you are trying to complete—not
    around the internal file structure of the package.
  </p>

  <div class="pygx-paths">
    <a class="pygx-path" href="getting-started/">
      <span class="pygx-path__number">01 · LEARN</span>
      <h3>Build the foundations</h3>
      <p>Understand spatial weights, kernels, bandwidths, coordinates, local calibration, and result interpretation.</p>
      <span class="pygx-path__arrow">Getting started →</span>
    </a>
    <a class="pygx-path" href="models/">
      <span class="pygx-path__number">02 · SELECT</span>
      <h3>Choose the right model</h3>
      <p>Compare 19 model families by response type, spatial scale, temporal structure, prediction support, and assumptions.</p>
      <span class="pygx-path__arrow">Model handbook →</span>
    </a>
    <a class="pygx-path" href="examples/">
      <span class="pygx-path__number">03 · BUILD</span>
      <h3>Run complete workflows</h3>
      <p>Use 45 maintained scripts covering every model, every public function group, diagnostics, plotting, and I/O.</p>
      <span class="pygx-path__arrow">Runnable examples →</span>
    </a>
    <a class="pygx-path" href="api/">
      <span class="pygx-path__number">04 · REFERENCE</span>
      <h3>Inspect the public contract</h3>
      <p>Browse signatures, parameters, return objects, docstrings, source links, and mapped example code for 174 APIs.</p>
      <span class="pygx-path__arrow">API reference →</span>
    </a>
  </div>
</section>

<section class="pygx-section">
  <span class="pygx-section__kicker">A coherent toolkit</span>
  <h2>One workflow across five layers</h2>
  <p class="pygx-section__intro">
    Models share a consistent research workflow: fit, inspect, diagnose, visualize,
    and report—with capability boundaries documented rather than hidden.
  </p>

  <div class="pygx-split">
    <div class="pygx-layer-list">
      <div class="pygx-layer"><span class="pygx-layer__index">01</span><div><h3>Models</h3><p>Regression, classification, transformation, local statistics, and inference families.</p></div></div>
      <div class="pygx-layer"><span class="pygx-layer__index">02</span><div><h3>Core numerics</h3><p>Kernels, distances, local solvers, bandwidth search, metrics, optimization, and validation.</p></div></div>
      <div class="pygx-layer"><span class="pygx-layer__index">03</span><div><h3>Diagnostics</h3><p>Residuals, influence, inference, local collinearity, temporal structure, weights, and regimes.</p></div></div>
      <div class="pygx-layer"><span class="pygx-layer__index">04</span><div><h3>Visualization</h3><p>Fifty-six model-aware and array-based plotting functions returning Matplotlib objects.</p></div></div>
      <div class="pygx-layer"><span class="pygx-layer__index">05</span><div><h3>I/O and examples</h3><p>NumPy/pandas-first contracts, built-in GeoDataFrame integration, persistence, and reproducible examples.</p></div></div>
    </div>

    <div class="pygx-code-card">
      <div class="pygx-code-card__top">
        <span class="pygx-code-card__dots"><i></i><i></i><i></i></span>
        <span>quickstart.py</span>
      </div>

<div class="highlight"><pre><code class="language-python">from pygwrx import GWR
from pygwrx.diagnostics import diagnostics_frame
from pygwrx.plotting import plot_diagnostic_panel

model = GWR(
    kernel=&quot;bisquare&quot;,
    bandwidth=48,
    adaptive=True,
)
model.fit(X, y, coords)

print(model.summary())
print(diagnostics_frame([model], labels=[&quot;GWR&quot;]))

fig, axes = plot_diagnostic_panel(model, theme=&quot;paper&quot;)
fig.savefig(&quot;gwr_diagnostics.png&quot;, dpi=200)</code></pre></div>
    </div>
  </div>
</section>

<section class="pygx-section">
  <span class="pygx-section__kicker">Model landscape</span>
  <h2>Start from the scientific structure</h2>
  <p class="pygx-section__intro">
    Begin with the simplest model that matches the response and spatial process,
    then add complexity only when diagnostics and theory justify it.
  </p>

  <div class="pygx-model-groups">
    <article class="pygx-model-group">
      <h3>Core local regression</h3>
      <p>Continuous responses with spatially varying relationships.</p>
      <div class="pygx-model-links"><a href="models/gwr/">GWR</a><a href="models/mgwr/">MGWR</a><a href="models/mixed-gwr/">MixedGWR</a></div>
    </article>
    <article class="pygx-model-group">
      <h3>Robust and regularized</h3>
      <p>Outliers, instability, local collinearity, and sparse effects.</p>
      <div class="pygx-model-links"><a href="models/rgwr/">RGWR</a><a href="models/lcr-gwr/">LCRGWR</a><a href="models/gw-lasso/">GWLasso</a></div>
    </article>
    <article class="pygx-model-group">
      <h3>Generalized and categorical</h3>
      <p>Counts, binary outcomes, and locally varying classes.</p>
      <div class="pygx-model-links"><a href="models/gwglm/">GWGLM</a><a href="models/gwda/">GWDA</a></div>
    </article>
    <article class="pygx-model-group">
      <h3>Space and time</h3>
      <p>Row-wise timestamps, historical stages, and parameter-specific scales.</p>
      <div class="pygx-model-links"><a href="models/gtwr/">GTWR</a><a href="models/stwr/">STWR</a><a href="models/sgtwr/">SGTWR</a><a href="models/mgtwr/">MGTWR</a></div>
    </article>
    <article class="pygx-model-group">
      <h3>Multivariate and scalable</h3>
      <p>Local structure, descriptive statistics, inference, and larger samples.</p>
      <div class="pygx-model-links"><a href="models/gwpca/">GWPCA</a><a href="models/gwss/">GWSS</a><a href="models/bootstrap-gwr/">BootstrapGWR</a><a href="models/scalable-gwr/">ScalableGWR</a></div>
    </article>
    <article class="pygx-model-group">
      <h3>Similarity and research models</h3>
      <p>Functional neighbourhoods, latent geometry, and connected regimes.</p>
      <div class="pygx-model-links"><a href="models/sgwr/">SGWR</a><a href="models/lg-gwr/">LGGWR</a><a href="models/gr-gwr/">GRGWR</a></div>
    </article>
  </div>
</section>

<section class="pygx-section">
  <span class="pygx-section__kicker">Research quality</span>
  <h2>Designed to make assumptions visible</h2>
  <div class="pygx-proof">
    <article><h3>Explicit capability boundaries</h3><p>Prediction, transformation, classification, statistics, and inference are documented as different operations—not presented as interchangeable estimators.</p></article>
    <article><h3>Examples as a public contract</h3><p>Every public API is mapped to a maintained example, with automated 174/174 coverage validation.</p></article>
    <article><h3>Reproducible engineering</h3><p>Deterministic example data, strict documentation builds, typed result objects, isolated optional dependencies, and reference comparisons where available.</p></article>
  </div>
</section>

<section class="pygx-cta">
  <div>
    <h2>Ready to fit your first local model?</h2>
    <p>Install the base package, run the five-minute workflow, then use diagnostics before adding model complexity.</p>
  </div>
  <a class="md-button" href="getting-started/installation/">Install pyGWRx</a>
</section>

</div>
