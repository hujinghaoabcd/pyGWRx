# API stability and design contract

pyGWRx 0.1.x is an Alpha spatial-statistics library. Public model names and core
fit/predict/result conventions are maintained deliberately, but the package does
not implement or promise the scikit-learn estimator protocol. In particular,
`clone`, `Pipeline`, `GridSearchCV`, `get_params`, and `set_params` are outside the
public compatibility contract.

Model APIs follow their statistical task: regressors expose fit/predict/result
methods, transformers expose transform-oriented methods, and diagnostics expose
analysis-oriented outputs. Fitted `summary()` methods return printable plain-text
reports; machine-readable results remain available through attributes, result
objects, `to_frame()`, and related export methods.
