#' @include classes.R getset.R generics.R
NULL

#################
###           ###
### Construct ###
###           ###
#################

#' @param response_distribution Which response distribution to use
#' @param fixed_effects A data.frame
#'
#' @noRd
setMethod(
    f = "initialize",
    signature = "observation_parameters",
    definition = function(
      .Object,
      response_parameters = list(
        removals = data.frame(
          par = numeric(2),
          se = numeric(2),
          fixed = logical(2),
          row.names = c("catchability", "sd")
        ),
        effort = data.frame(
          par = numeric(1),
          se = numeric(1),
          fixed = logical(1),
          row.names = c("sd")
        )
      ),
      fixed_effects = list(
        biomass = data.frame(
          par = numeric(0),
          se = numeric(0),
          fixed = numeric(0)
        ),
        effort = data.frame(
          par = numeric(0),
          se = numeric(0),
          fixed = numeric(0)
        ),
        suitability = data.frame(
          par = numeric(0),
          se = numeric(0),
          fixed = numeric(0)
        )
      )) {
  response_parameters(.Object)<- response_parameters
  fixed_effects(.Object)<- fixed_effects

  return(.Object)
})


##############
###        ###
### Access ###
###        ###
##############


#' @param x An object
#'
#' @export
#' @describeIn observation_parameters_class Get response distribution parameters
setMethod(
    f = "response_parameters",
    signature = "observation_parameters",
    definition = function(x) {
  return(x@response_parameters)
})

#' @param x An object
#' @param value A replacement value
#'
#' @export
#' @describeIn observation_parameters_class Set response distribution parameters
setReplaceMethod(
    f = "response_parameters",
    signature = "observation_parameters",
    definition = function(x, value) {
  x@response_parameters<- value
  return(x)
})




#' @param x An object
#'
#' @export
#' @describeIn observation_parameters_class Get fixed effect parameters
setMethod(
    f = "fixed_effects",
    signature = "observation_parameters",
    definition = function(x) {
  return(x@fixed_effects)
})

#' @param x An object
#' @param value A replacement value
#'
#' @export
#' @describeIn observation_parameters_class Set fixed effect parameters
setReplaceMethod(
    f = "fixed_effects",
    signature = "observation_parameters",
    definition = function(x, value) {
  x@fixed_effects<- value
  return(x)
})
