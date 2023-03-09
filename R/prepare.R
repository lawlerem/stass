#' @include classes.R getset.R generics.R process.R observations.R settings.R tracing.R TMB_out.R stass.R
NULL


#' Create an object of class \code{stass}.
#'
#' 'stass_prepare' is used to take an existing `simple features` data.frame
#'   with point geometries, time information, covariates, and age and length
#'   variables and perform all of the pre-processing steps necessary to fit a
#'   model with the \code{stass_fit} function. See the description for
#'   \link{stass_prepare_process} and \link{stass_prepare_observations} for more
#'   details on how each part is prepared.
#'
#' The formula object should always be of the form \code{catch ~ effort(e) +
#'   x + z + time(t, type = "ar1") + space("matern", nu = 1.5)},
#'   though possibly with some terms missing.
#'
#' The \code{time(...)} term indicates which column, if any, holds the time
#'   index. The variable t should be replaced with the desired time index. There
#'   are currently three valid options for the `type' argument in
#'   \code{time(t,type="ar1")} -- "ar1" for an AR(1) structure, "rw" for a
#'   random walk, and "independent" for independent spatial fields each year. If
#'   the \code{time(...)} term is missing, all observations are assumed to be
#'   at the same time and a purely spatial model is used.
#'
#' The \code{space(...)} term specifies the spatial covariance function. See
#'   \code{get_starve_distributions("covariance")} for valid names to supply.
#'   If using the "matern" option you can supply a value for the smoothness
#'   parameter nu, which will be held constant in model fitting. If nu is not
#'   given, then it will be freely estimated in the model. If the
#'   \code{space(...)} term as a whole is missing, an exponential covariance
#'   function is assumed.
#'
#' @param formula A formula object. See the 'Details' section below.
#' @param data An `sf` object containing point geometries, and any other
#'   variables needed to fit the model.
#' @param mesh An INLA mesh object. See INLA::inla.mesh.2d.
#' @param silent Logical. Should intermediate calculations be printed?
#' @param distance_units Any value that can be used as a \code{units} object
#'   from the \code{units} package. Which distance units should the model use?
#'   Defaults to "km".
#' @param fit Logical (Default=FALSE). Should parameter estimates be found?
#'   If so, the starting values for the optimizer will use the default values.
#' @param ... Extra options to pass to \link{stass_fit} if fit=TRUE
#'
#' @return An stass object. If fit=TRUE, the returned model parameters will be
#'   estimated using the \link{stass_fit} function using the default starting
#'   values.
#'
#' @seealso stass_class
#'
#' @export
stass_prepare<- function(
    formula,
    data,
    mesh,
    silent = TRUE,
    distance_units = "km",
    fit = FALSE,
    ...) {
  model<- new("stass")

  # Set the settings in the model
  settings(model)<- new(
    "settings",
    formula = formula,
    distance_units = distance_units
  )

  # Set up the process
  process(model)<- stass_prepare_process(
    data = data,
    mesh = mesh,
    settings = settings(model)
  )

  # Set up the observations
  observations(model)<- stass_prepare_observations(
    data = data,
    process = process(model),
    settings = settings(model)
  )

  if( fit ) {
    model<- stass_fit(model, silent = silent, ...)
  } else {}

  return(model)
}



#' @param settings A settings object
#'
#' @describeIn stass_prepare Creates a new process object with the correct
#'   dimensions for the temporal random effects, persistent graph random
#'   effects, and transient graph random effects. Initializes the temporal and
#'   spatial parameters for the model according to the options specified in the
#'   formula element of the settings argument. Constructs the persistent and
#'   transient graph, see \link{construct_graph}.
#'
#' Before creating the transient graph any location in data that
#'   is present in the mesh is removed.
stass_prepare_process<- function(
    data,
    mesh,
    settings) {
  process<- new("process")

  # Return a time column with name and type (ar1/rw/etc) attributes
  time_col<- time_from_formula(formula(settings), data)
  time_seq<- seq(min(time_col), max(time_col))
  sf_name<- attr(data, "sf_column")



  # time_effects = "data.frame"
  time_effects(process)<- stars::st_as_stars(
    list(
      w = array(
        1,
        dim = c(
          length(time_seq),
          3
        )
      ),
      se = array(
        0,
        dim = c(
          length(time_seq),
          3
        )
      )
    ),
    dimensions = stars::st_dimensions(
      time = time_seq,
      variable = c("biomass", "effort", "suitability")
    )
  )
  names(stars::st_dimensions(time_effects(process)))[[1]]<- time_name(settings)


  # pg_re = "sf"
  graph<- construct_dag(
    mesh,
    settings = settings
  )
  uniq_nodes<- graph$locations
  persistent_graph(process)<- graph$dag

  pg_re(process)<- stars::st_as_stars(
    list(
      w = array(
        1,
        dim = c(
          nrow(uniq_nodes),
          length(time_seq),
          3
        )
      ),
      se = array(
        0,
        dim = c(
          nrow(uniq_nodes),
          length(time_seq),
          3
        )
      )
    ),
    dimensions = stars::st_dimensions(
      geom = sf::st_geometry(uniq_nodes),
      time = time_seq,
      variable = c("biomass", "effort", "suitability")
    )
  )
  names(stars::st_dimensions(pg_re(process)))[[2]]<- time_name(settings)


  # tg_re = "sf"
  # Construct transient graph if not supplied
  pg_locs<- locations_from_stars(pg_re(process))
  colnames(pg_locs)[colnames(pg_locs) == attr(pg_locs,"sf_column")]<- sf_name
  sf::st_geometry(pg_locs)<- sf_name
  data<- data[
    !duplicated(
      data[
        ,
        c(time_name(settings), attr(data, "sf_column"))
      ]
    ),
  ]
  data<- data[lengths(sf::st_equals(data, pg_locs)) == 0, ]

  transient_graph(process)<- construct_transient_graph(
    x = data,
    y = pg_locs,
    time = time_from_formula(formula(settings), data),
    settings = settings
  )
  tg_re(process)<- new(
    "long_stars",
    locations = sf::st_sf(
      time_from_formula(
        formula(settings),
        data,
        return = "column"),
      geom = sf::st_geometry(data)
    ),
    var_names = c("max", "rate")
  )
  names(values(tg_re(process)))[[2]]<- "se"
  values(tg_re(process))$w[]<- 1
  values(tg_re(process))$linear<- NULL
  values(tg_re(process))$linear_se<- NULL
  values(tg_re(process))$response<- NULL
  values(tg_re(process))$response_se<- NULL



  # parameters = "process_parameters"
  parameters<- new("process_parameters")
  # Returns name of covariance function and value of nu
  covariance<- covariance_from_formula(formula(settings))

  covariance_function(parameters)<- covariance$covariance
  # covariance_function<- takes care of settings spatial parameters,
  # but if nu is supplied for matern need to set nu
  for( i in seq_along(covariance$nu) ) {
    if( covariance$covariance[[i]] == "matern" && !is.na(covariance$nu[[i]]) ) {
      space_parameters(parameters)[[i]]["nu", "par"]<- covariance$nu[[i]]
      space_parameters(parameters)[[i]]["nu", "fixed"]<- TRUE
    } else {}
  }
  names(space_parameters(parameters))<- c("max", "rate")

  time_parameters(parameters)<- list(
    biomass = data.frame(
      par = c(0, 0),
      se = c(0, 0),
      fixed = c(FALSE, FALSE)
      row.names = c("mu", "sd")
    ),
    effort = data.frame(
      par = c(0, 0, 0, 0),
      se = c(0, 0, 0, 0),
      fixed = c(FALSE, FALSE, FALSE, FALSE),
      row.names = c("mu", "sd", "chi", "alpha")
    ),
    suitability = data.frame(
      par = c(0, 0),
      se = c(0, 0),
      fixed = c(FALSE, FALSE),
      row.names = c("mu", "sd")
    )
  )
  if( length(unique(time_seq)) == 1 ) {
    time_parameters(parameters)$biomass["sd", "fixed"]<- TRUE
    time_parameters(parameters)$effort[c("sd", "chi", "alpha"), "fixed"]<- TRUE
    time_parameters(parameters)$suitability["sd", "fixed"]<- TRUE
  } else {}
  
  parameters(process)<- parameters

  return(process)
}




#' @param process A process object.
#'
#' @describeIn stass_prepare Creates a new observation object with the correct dimensions
#'   for the random effect predictions. Initializes the response distribution
#'   and fixed effect parameters for the model according to the options specified
#'   in the formula element of the settings argument. Also adds a column "graph_idx"
#'   to the supplied data.
stass_prepare_observations<- function(
    data,
    process,
    settings) {
  observations<- new("observations")

  # data = "sf"
  # Put in lexicographic ordering by time, then S->N / W->E
  data<- order_by_location(data, time = data[[time_name(settings)]])

  data_predictions(observations)<- new(
    "long_stars",
    sf::st_sf(
      data.frame(
        mean_design_from_formula(formula(settings), data, return = "all.vars"),
        response_from_formula(formula(settings), data),
        effort_from_formula(formula(settings), data),
        time_from_formula(formula(settings), data, "column"),
        graph_idx = create_graph_idx(data, process, settings),
        data[, attr(data, "sf_column")]
      )
    ),
    var_names = c("biomass", "effort", "suitability")
  )



  # parameters = "observation_parameters"
  parameters<- new("observation_parameters")

  # Set up fixed effects according to covariates formula
  nff<- colnames(mean_design_from_formula(formula(settings), data))
  fixed_effects(parameters)<- lapply(
    c("biomass", "effort", "suitability"),
    function(rr) {
      return(data.frame(
        par = numeric(length(nff)),
        se = rep(NA, length(nff)),
        fixed = rep(FALSE, length(nff)),
        row.names = nff
      ))
    }
  )
  names(fixed_effects(parameters))<- c("biomass", "effort", "rate")
  parameters(observations)<- parameters

  return(observations)
}
