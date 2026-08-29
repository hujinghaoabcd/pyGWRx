#!/usr/bin/env Rscript

# Generate GWmodel/spgwr references for the real Columbus GWR dataset.

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
source_path <- file.path(root, "src", "pygwrx", "data", "Columbus", "columbus.csv")
output_dir <- file.path(root, "tests", "reference_data", "gwr", "real_columbus")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

frame <- read.csv(source_path, check.names = FALSE)
coords <- as.matrix(frame[, c("X", "Y")])
formula <- CRIME ~ INC + HOVAL
spdf <- SpatialPointsDataFrame(
  coords,
  data = frame[, c("CRIME", "INC", "HOVAL")],
  match.ID = FALSE
)

# Zero-based Python rows 0,10,20,30,40 correspond to these R rows.
holdout_rows <- c(1L, 11L, 21L, 31L, 41L)
holdout <- frame[holdout_rows, , drop = FALSE]
training <- frame[-holdout_rows, , drop = FALSE]
training_coords <- as.matrix(training[, c("X", "Y")])
holdout_coords <- as.matrix(holdout[, c("X", "Y")])
training_spdf <- SpatialPointsDataFrame(
  training_coords,
  data = training[, c("CRIME", "INC", "HOVAL")],
  match.ID = FALSE
)
holdout_spdf <- SpatialPointsDataFrame(
  holdout_coords,
  data = holdout[, c("INC", "HOVAL")],
  match.ID = FALSE
)

matrix_payload <- function(x) {
  if (is.null(x)) return(NULL)
  unname(as.matrix(x))
}

numeric_payload <- function(x) {
  if (is.null(x)) return(NULL)
  unname(as.numeric(x))
}

get_column <- function(data, candidates) {
  for (candidate in candidates) {
    if (candidate %in% names(data)) return(numeric_payload(data[[candidate]]))
  }
  NULL
}

get_columns <- function(data, suffix) {
  columns <- grep(paste0(suffix, "$"), names(data), value = TRUE)
  if (length(columns) == 0L) return(NULL)
  matrix_payload(data[, columns, drop = FALSE])
}

named_numeric_list <- function(x) {
  if (is.null(x)) return(list())
  out <- list()
  for (name in names(x)) {
    value <- x[[name]]
    if (is.numeric(value) && length(value) == 1L && is.finite(value)) {
      out[[name]] <- as.numeric(value)
    }
  }
  out
}

gwmodel_case <- function(name, kernel, bandwidth, adaptive) {
  fit <- gwr.basic(
    formula,
    data = spdf,
    bw = bandwidth,
    kernel = kernel,
    adaptive = adaptive,
    p = 2,
    theta = 0,
    longlat = FALSE,
    F123.test = FALSE,
    cv = FALSE
  )
  sdf <- fit$SDF@data
  list(
    config = list(
      name = name,
      kernel = kernel,
      bandwidth = bandwidth,
      adaptive = adaptive,
      sigma2_convention = "v2"
    ),
    params = matrix_payload(sdf[, seq_len(3L), drop = FALSE]),
    predy = get_column(sdf, c("yhat", "pred")),
    residuals = get_column(sdf, c("residual", "gwr.e")),
    local_r2 = get_column(sdf, c("Local_R2", "localR2")),
    bse = get_columns(sdf, "_SE"),
    tvalues = get_columns(sdf, "_TV"),
    diagnostics = named_numeric_list(fit$GW.diagnostic),
    output_columns = names(sdf)
  )
}

gwmodel_holdout_prediction <- function() {
  fit <- gwr.basic(
    formula,
    data = training_spdf,
    regression.points = holdout_spdf,
    bw = 10.0,
    kernel = "gaussian",
    adaptive = FALSE,
    p = 2,
    theta = 0,
    longlat = FALSE,
    F123.test = FALSE,
    cv = FALSE
  )
  sdf <- fit$SDF@data
  params <- as.matrix(sdf[, seq_len(3L), drop = FALSE])
  design <- cbind(1.0, as.matrix(holdout[, c("INC", "HOVAL")]))
  list(
    config = list(kernel = "gaussian", bandwidth = 10.0, fixed = TRUE),
    holdout_rows_zero_based = as.integer(holdout_rows - 1L),
    holdout_polyid = as.integer(holdout$POLYID),
    actual_response = numeric_payload(holdout$CRIME),
    coords = matrix_payload(holdout_coords),
    X = matrix_payload(holdout[, c("INC", "HOVAL")]),
    params = matrix_payload(params),
    predictions = numeric_payload(rowSums(design * params)),
    n_training = nrow(training),
    n_holdout = nrow(holdout)
  )
}

spgwr_data <- function(fit) {
  if (inherits(fit$SDF, "Spatial")) return(fit$SDF@data)
  as.data.frame(fit$SDF)
}

spgwr_params <- function(data) {
  required <- c("(Intercept)", "INC", "HOVAL")
  if (!all(required %in% names(data))) {
    stop("spgwr output does not contain the expected coefficient columns.")
  }
  matrix_payload(data[, required, drop = FALSE])
}

spgwr_case <- function(name, kernel_function, bandwidth = NULL, adapt = NULL) {
  arguments <- list(
    formula = formula,
    data = frame,
    coords = coords,
    gweight = kernel_function,
    se.fit = TRUE,
    se.fit.CCT = TRUE,
    predictions = TRUE,
    longlat = FALSE
  )
  if (!is.null(bandwidth)) arguments$bandwidth <- bandwidth
  if (!is.null(adapt)) arguments$adapt <- adapt
  fit <- do.call(spgwr::gwr, arguments)
  sdf <- spgwr_data(fit)
  list(
    config = list(name = name, bandwidth = bandwidth, adapt = adapt),
    params = spgwr_params(sdf),
    predy = get_column(sdf, c("pred", "yhat")),
    residuals = get_column(sdf, c("gwr.e", "residual")),
    local_r2 = get_column(sdf, c("localR2", "Local_R2")),
    bse = get_columns(sdf, "_se"),
    diagnostics = named_numeric_list(fit$results),
    output_columns = names(sdf)
  )
}

spgwr_holdout_prediction <- function() {
  arguments <- list(
    formula = formula,
    data = training,
    coords = training_coords,
    bandwidth = 10.0,
    gweight = spgwr::gwr.Gauss,
    fit.points = holdout_spdf,
    predictions = TRUE,
    se.fit = FALSE,
    longlat = FALSE
  )
  tryCatch({
    fit <- do.call(spgwr::gwr, arguments)
    sdf <- spgwr_data(fit)
    params <- as.matrix(spgwr_params(sdf))
    design <- cbind(1.0, as.matrix(holdout[, c("INC", "HOVAL")]))
    list(
      config = list(kernel = "gaussian", bandwidth = 10.0, fixed = TRUE),
      holdout_rows_zero_based = as.integer(holdout_rows - 1L),
      holdout_polyid = as.integer(holdout$POLYID),
      actual_response = numeric_payload(holdout$CRIME),
      coords = matrix_payload(holdout_coords),
      X = matrix_payload(holdout[, c("INC", "HOVAL")]),
      params = matrix_payload(params),
      predictions = numeric_payload(rowSums(design * params)),
      output_columns = names(sdf),
      n_training = nrow(training),
      n_holdout = nrow(holdout)
    )
  }, error = function(error) {
    list(error = conditionMessage(error))
  })
}

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

# Controlled GWmodel criterion curve on exactly the same integer k=4..49
# candidates used by pyGWRx and mgwr.
mf <- model.frame(formula, data = frame)
X_design <- model.matrix(attr(mf, "terms"), mf)
Y_response <- model.response(mf)
candidates <- 4:49
dMat <- gw.dist(
  dp.locat = coords,
  rp.locat = coords,
  p = 2,
  theta = 0,
  longlat = FALSE
)
gw_cv <- get("gwr.cv", envir = asNamespace("GWmodel"))
gw_aicc <- get("gwr.aic", envir = asNamespace("GWmodel"))
gw_bic <- get("gwr.bic", envir = asNamespace("GWmodel"))

gwmodel_curve_points <- lapply(candidates, function(k) {
  cv <- safe_eval(gw_cv(
    bw = k,
    X = X_design,
    Y = Y_response,
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
    X = X_design,
    Y = Y_response,
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
    X = X_design,
    Y = Y_response,
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
  generator = "tools/reference/gwr/generate_columbus_r_references.R",
  reference_package = "GWmodel",
  reference_version = as.character(packageVersion("GWmodel")),
  dataset = "Columbus (OH) neighborhood crime",
  dataset_source = "src/pygwrx/data/Columbus/columbus.csv",
  formula = "CRIME ~ INC + HOVAL",
  n_samples = nrow(frame),
  features = c("INC", "HOVAL"),
  response = "CRIME",
  coords = c("X", "Y"),
  cases = list(
    fixed_gaussian_v2 = gwmodel_case("fixed_gaussian_v2", "gaussian", 10.0, FALSE),
    fixed_bisquare_v2 = gwmodel_case("fixed_bisquare_v2", "bisquare", 15.0, FALSE),
    adaptive_gaussian_v2 = gwmodel_case("adaptive_gaussian_v2", "gaussian", 24, TRUE),
    adaptive_bisquare_v2 = gwmodel_case("adaptive_bisquare_v2", "bisquare", 24, TRUE)
  ),
  held_out_fixed_gaussian_prediction = gwmodel_holdout_prediction(),
  adaptive_bisquare_bandwidth_curve = list(
    implementation = "GWmodel",
    candidate_semantics = "adaptive integer nearest-neighbour bandwidth",
    candidate_min = min(candidates),
    candidate_max = max(candidates),
    kernel = "bisquare",
    points = gwmodel_curve_points
  )
)

spgwr_payload <- list(
  generator = "tools/reference/gwr/generate_columbus_r_references.R",
  reference_package = "spgwr",
  reference_version = as.character(packageVersion("spgwr")),
  dataset = "Columbus (OH) neighborhood crime",
  dataset_source = "src/pygwrx/data/Columbus/columbus.csv",
  formula = "CRIME ~ INC + HOVAL",
  n_samples = nrow(frame),
  features = c("INC", "HOVAL"),
  response = "CRIME",
  coords = c("X", "Y"),
  notes = list(
    gaussian = "Uses gwr.Gauss, not the deprecated gwr.gauss definition.",
    adaptive = "spgwr uses a sample proportion q; q=24/49 is a semantic cross-check, not exact integer-k equivalence."
  ),
  cases = list(
    fixed_gaussian = spgwr_case("fixed_gaussian", spgwr::gwr.Gauss, bandwidth = 10.0),
    fixed_bisquare = spgwr_case("fixed_bisquare", spgwr::gwr.bisquare, bandwidth = 15.0),
    adaptive_gaussian = spgwr_case("adaptive_gaussian", spgwr::gwr.Gauss, adapt = 24 / 49),
    adaptive_bisquare = spgwr_case("adaptive_bisquare", spgwr::gwr.bisquare, adapt = 24 / 49)
  ),
  held_out_fixed_gaussian_prediction = spgwr_holdout_prediction()
)

write_json(
  gwmodel_payload,
  file.path(output_dir, "GWmodel_reference.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(output_dir, "GWmodel_reference.json"), append = TRUE)
write_json(
  spgwr_payload,
  file.path(output_dir, "spgwr_reference.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(output_dir, "spgwr_reference.json"), append = TRUE)

cat("wrote tests/reference_data/gwr/real_columbus/GWmodel_reference.json\n")
cat("wrote tests/reference_data/gwr/real_columbus/spgwr_reference.json\n")
