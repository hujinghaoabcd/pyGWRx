#!/usr/bin/env Rscript

# Generate frozen GWR references with GWmodel and spgwr.

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
prediction_frame <- read.csv(file.path(data_dir, "prediction.csv"), check.names = FALSE)
coords <- as.matrix(frame[, c("x", "ycoord")])
prediction_coords <- as.matrix(prediction_frame[, c("x", "ycoord")])
formula <- response ~ x1 + x2
spdf <- SpatialPointsDataFrame(
  coords,
  data = frame[, c("response", "x1", "x2")],
  match.ID = FALSE
)
prediction_spdf <- SpatialPointsDataFrame(
  prediction_coords,
  data = prediction_frame[, c("x1", "x2")],
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
    hatmatrix = TRUE,
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

gwmodel_prediction <- function() {
  fit <- gwr.basic(
    formula,
    data = spdf,
    regression.points = prediction_spdf,
    bw = 55.0,
    kernel = "gaussian",
    adaptive = FALSE,
    p = 2,
    theta = 0,
    longlat = FALSE,
    hatmatrix = FALSE,
    F123.test = FALSE,
    cv = FALSE
  )
  sdf <- fit$SDF@data
  params <- as.matrix(sdf[, seq_len(3L), drop = FALSE])
  design <- cbind(1.0, as.matrix(prediction_frame[, c("x1", "x2")]))
  list(
    coords = matrix_payload(prediction_coords),
    X = matrix_payload(prediction_frame[, c("x1", "x2")]),
    params = matrix_payload(params),
    predictions = numeric_payload(rowSums(design * params))
  )
}

gwmodel_selected_bandwidths <- function() {
  list(
    cv = as.numeric(bw.gwr(
      formula,
      data = spdf,
      approach = "CV",
      kernel = "bisquare",
      adaptive = TRUE,
      p = 2,
      theta = 0,
      longlat = FALSE
    )),
    aicc = as.numeric(bw.gwr(
      formula,
      data = spdf,
      approach = "AICc",
      kernel = "bisquare",
      adaptive = TRUE,
      p = 2,
      theta = 0,
      longlat = FALSE
    ))
  )
}

spgwr_data <- function(fit) {
  if (inherits(fit$SDF, "Spatial")) return(fit$SDF@data)
  as.data.frame(fit$SDF)
}

spgwr_case <- function(name, kernel_function, bandwidth = NULL, adapt = NULL) {
  arguments <- list(
    formula = formula,
    data = frame,
    coords = coords,
    gweight = kernel_function,
    hatmatrix = TRUE,
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
    config = list(
      name = name,
      bandwidth = bandwidth,
      adapt = adapt
    ),
    params = matrix_payload(sdf[, seq_len(3L), drop = FALSE]),
    predy = get_column(sdf, c("pred", "yhat")),
    residuals = get_column(sdf, c("gwr.e", "residual")),
    local_r2 = get_column(sdf, c("localR2", "Local_R2")),
    bse = get_columns(sdf, "_se"),
    diagnostics = named_numeric_list(fit$results),
    output_columns = names(sdf)
  )
}

spgwr_prediction <- function() {
  arguments <- list(
    formula = formula,
    data = frame,
    coords = coords,
    bandwidth = 55.0,
    gweight = spgwr::gwr.Gauss,
    fit.points = prediction_spdf,
    predictions = TRUE,
    hatmatrix = FALSE,
    se.fit = FALSE,
    longlat = FALSE
  )
  tryCatch({
    fit <- do.call(spgwr::gwr, arguments)
    sdf <- spgwr_data(fit)
    params <- as.matrix(sdf[, seq_len(3L), drop = FALSE])
    design <- cbind(1.0, as.matrix(prediction_frame[, c("x1", "x2")]))
    list(
      coords = matrix_payload(prediction_coords),
      X = matrix_payload(prediction_frame[, c("x1", "x2")]),
      params = matrix_payload(params),
      predictions = numeric_payload(rowSums(design * params)),
      output_columns = names(sdf)
    )
  }, error = function(error) {
    list(error = conditionMessage(error))
  })
}

spgwr_selected_bandwidths <- function() {
  list(
    cv_fixed = as.numeric(gwr.sel(
      formula,
      data = frame,
      coords = coords,
      gweight = spgwr::gwr.bisquare,
      method = "cv",
      verbose = FALSE,
      longlat = FALSE
    )),
    aic_fixed = as.numeric(gwr.sel(
      formula,
      data = frame,
      coords = coords,
      gweight = spgwr::gwr.bisquare,
      method = "aic",
      verbose = FALSE,
      longlat = FALSE
    ))
  )
}

gwmodel_payload <- list(
  generator = "tools/reference/gwr/generate_r_references.R",
  reference_package = "GWmodel",
  reference_version = as.character(packageVersion("GWmodel")),
  n_samples = nrow(frame),
  features = c("x1", "x2"),
  cases = list(
    fixed_gaussian_v2 = gwmodel_case("fixed_gaussian_v2", "gaussian", 55.0, FALSE),
    fixed_bisquare_v2 = gwmodel_case("fixed_bisquare_v2", "bisquare", 70.0, FALSE),
    adaptive_gaussian_v2 = gwmodel_case("adaptive_gaussian_v2", "gaussian", 20, TRUE),
    adaptive_bisquare_v2 = gwmodel_case("adaptive_bisquare_v2", "bisquare", 20, TRUE)
  ),
  adaptive_bisquare_bandwidth_selection = gwmodel_selected_bandwidths(),
  fixed_gaussian_prediction = gwmodel_prediction()
)

spgwr_payload <- list(
  generator = "tools/reference/gwr/generate_r_references.R",
  reference_package = "spgwr",
  reference_version = as.character(packageVersion("spgwr")),
  n_samples = nrow(frame),
  features = c("x1", "x2"),
  notes = list(
    gaussian = "Uses gwr.Gauss, not the deprecated gwr.gauss definition.",
    adaptive = "spgwr expresses adaptive bandwidth as a sample proportion; 20/40 = 0.5."
  ),
  cases = list(
    fixed_gaussian = spgwr_case("fixed_gaussian", spgwr::gwr.Gauss, bandwidth = 55.0),
    fixed_bisquare = spgwr_case("fixed_bisquare", spgwr::gwr.bisquare, bandwidth = 70.0),
    fixed_tricube = spgwr_case("fixed_tricube", spgwr::gwr.tricube, bandwidth = 70.0),
    adaptive_gaussian = spgwr_case("adaptive_gaussian", spgwr::gwr.Gauss, adapt = 0.5),
    adaptive_bisquare = spgwr_case("adaptive_bisquare", spgwr::gwr.bisquare, adapt = 0.5)
  ),
  fixed_bandwidth_selection = spgwr_selected_bandwidths(),
  fixed_gaussian_prediction = spgwr_prediction()
)

write_json(
  gwmodel_payload,
  file.path(data_dir, "GWmodel_reference.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(data_dir, "GWmodel_reference.json"), append = TRUE)
write_json(
  spgwr_payload,
  file.path(data_dir, "spgwr_reference.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 16,
  null = "null"
)
cat("\n", file = file.path(data_dir, "spgwr_reference.json"), append = TRUE)

cat("wrote tests/reference_data/gwr/GWmodel_reference.json\n")
cat("wrote tests/reference_data/gwr/spgwr_reference.json\n")
