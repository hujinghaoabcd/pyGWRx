#!/usr/bin/env Rscript

# Generate controlled GWR bandwidth criterion curves with GWmodel and spgwr.
# GWmodel is evaluated on the exact integer k candidates 4:40.  spgwr uses an
# adaptive sample proportion q, so q=k/n is retained only as a semantic
# cross-check and is not treated as exact k-neighbour equivalence.

suppressPackageStartupMessages({
  library(jsonlite)
  library(sp)
  library(GWmodel)
  library(spgwr)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1L) stop("Unable to locate script path.")
script_path <- normalizePath(sub("^--file=", "", file_arg))
root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."))
data_dir <- file.path(root, "tests", "reference_data", "gwr")
frame <- read.csv(file.path(data_dir, "input.csv"), check.names = FALSE)
coords <- as.matrix(frame[, c("x", "ycoord")])
formula <- response ~ x1 + x2
mf <- model.frame(formula, data = frame)
X <- model.matrix(attr(mf, "terms"), mf)
Y <- model.response(mf)
n <- nrow(frame)
candidates <- 4:40

dMat <- gw.dist(
  dp.locat = coords,
  rp.locat = coords,
  p = 2,
  theta = 0,
  longlat = FALSE
)

finite_or_null <- function(value) {
  if (length(value) != 1L || !is.finite(value)) return(NULL)
  as.numeric(value)
}

safe_eval <- function(expr) {
  tryCatch(
    list(status = "ok", value = finite_or_null(force(expr))),
    error = function(error) list(status = "error", error = conditionMessage(error))
  )
}

# Use the exact criterion functions called internally by bw.gwr.  gwr.aic is
# GWmodel's AICc objective despite the historical function name.
gw_cv <- get("gwr.cv", envir = asNamespace("GWmodel"))
gw_aicc <- get("gwr.aic", envir = asNamespace("GWmodel"))
gw_bic <- get("gwr.bic", envir = asNamespace("GWmodel"))

gwmodel_points <- lapply(candidates, function(k) {
  cv <- safe_eval(gw_cv(
    bw = k,
    X = X,
    Y = Y,
    kernel = "bisquare",
    adaptive = TRUE,
    dp.locat = coords,
    p = 2,
    theta = 0,
    longlat = FALSE,
    dMat = dMat,
    verbose = FALSE,
    parallel.method = FALSE
  ))
  aicc <- safe_eval(gw_aicc(
    bw = k,
    X = X,
    Y = Y,
    kernel = "bisquare",
    adaptive = TRUE,
    dp.locat = coords,
    p = 2,
    theta = 0,
    longlat = FALSE,
    dMat = dMat,
    verbose = FALSE,
    parallel.method = FALSE
  ))
  bic <- safe_eval(gw_bic(
    bw = k,
    X = X,
    Y = Y,
    kernel = "bisquare",
    adaptive = TRUE,
    dp.locat = coords,
    p = 2,
    theta = 0,
    longlat = FALSE,
    dMat = dMat,
    verbose = FALSE,
    parallel.method = FALSE
  ))

  point <- list(
    k = as.integer(k),
    cv_sse = cv$value,
    aicc = aicc$value,
    bic = bic$value,
    status = if (cv$status == "ok" && aicc$status == "ok" && bic$status == "ok") "ok" else "partial"
  )
  errors <- c(
    if (!is.null(cv$error)) paste0("cv: ", cv$error),
    if (!is.null(aicc$error)) paste0("aicc: ", aicc$error),
    if (!is.null(bic$error)) paste0("bic: ", bic$error)
  )
  if (length(errors) > 0L) point$errors <- errors
  point
})

gwmodel_payload <- list(
  implementation = "GWmodel",
  reference_version = as.character(packageVersion("GWmodel")),
  candidate_semantics = "adaptive integer nearest-neighbour bandwidth",
  kernel = "bisquare",
  candidate_min = min(candidates),
  candidate_max = max(candidates),
  n_samples = n,
  criterion_notes = list(
    cv_sse = "GWmodel::gwr.cv; strict leave-one-out sum of squared errors",
    aicc = "GWmodel internal gwr.aic objective used by bw.gwr(approach='AICc')",
    bic = "GWmodel internal gwr.bic objective"
  ),
  points = gwmodel_points
)

# spgwr's adaptive selector is parameterized by a continuous sample proportion
# q rather than an integer neighbour order.  q=k/n is deliberately archived as
# a semantic sensitivity curve, not as a strict numerical reference.
sp_cv_adapt <- get("gwr.cv.adapt.f", envir = asNamespace("spgwr"))
sp_aic_adapt <- get("gwr.aic.adapt.f", envir = asNamespace("spgwr"))
case_weights <- rep(1.0, n)

spgwr_points <- lapply(candidates, function(k) {
  q <- k / n
  cv <- safe_eval(sp_cv_adapt(
    q = q,
    y = Y,
    x = X,
    coords = coords,
    gweight = spgwr::gwr.bisquare,
    verbose = FALSE,
    longlat = FALSE,
    RMSE = FALSE,
    weights = case_weights,
    show.error.messages = FALSE
  ))
  aic <- safe_eval(sp_aic_adapt(
    q = q,
    y = Y,
    x = X,
    coords = coords,
    gweight = spgwr::gwr.bisquare,
    verbose = FALSE,
    longlat = FALSE,
    show.error.messages = FALSE
  ))
  point <- list(
    k_equivalent = as.integer(k),
    q = q,
    cv_sse = cv$value,
    aicc_like = aic$value,
    status = if (cv$status == "ok" && aic$status == "ok") "ok" else "partial"
  )
  errors <- c(
    if (!is.null(cv$error)) paste0("cv: ", cv$error),
    if (!is.null(aic$error)) paste0("aic: ", aic$error)
  )
  if (length(errors) > 0L) point$errors <- errors
  point
})

spgwr_payload <- list(
  implementation = "spgwr",
  reference_version = as.character(packageVersion("spgwr")),
  candidate_semantics = "adaptive continuous sample proportion q; q=k/n shown only for semantic cross-check",
  kernel = "bisquare",
  candidate_min = min(candidates),
  candidate_max = max(candidates),
  n_samples = n,
  points = spgwr_points
)

write_json(
  gwmodel_payload,
  file.path(data_dir, "GWmodel_bandwidth_curve.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(data_dir, "GWmodel_bandwidth_curve.json"), append = TRUE)
write_json(
  spgwr_payload,
  file.path(data_dir, "spgwr_bandwidth_curve.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(data_dir, "spgwr_bandwidth_curve.json"), append = TRUE)

cat("wrote tests/reference_data/gwr/GWmodel_bandwidth_curve.json\n")
cat("wrote tests/reference_data/gwr/spgwr_bandwidth_curve.json\n")
