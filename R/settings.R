#' @include classes.R getset.R generics.R
NULL

#################
###           ###
### Construct ###
###           ###
#################

#' @param formula A formula
#' @param distance_units Which distance units to use
#'
#' @noRd
setMethod(
    f = "initialize",
    signature = "settings",
    definition = function(
      .Object,
      formula = new("formula"),
      distance_units = "km"
    ) {
  formula(.Object)<- formula
  .Object@distance_units<- distance_units

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
#' @describeIn settings_class Get model formula
setMethod(
    f = "formula",
    signature = "settings",
    definition = function(x) {
  return(x@formula)
})

#' @param x An object
#' @param value A replacement value
#'
#' @export
#' @describeIn settings_class Set model formula
setReplaceMethod(
    f = "formula",
    signature = "settings",
    definition = function(x, value) {
  x@formula<- value
  return(x)
})




#' @param x An object
#'
#' @export
#' @describeIn settings_class Get distance units used for the model.
setMethod(
    f = "distance_units",
    signature = "settings",
    definition = function(x) {
  return(x@distance_units)
})

#' @param x An object
#' @param value A replacement value
#'
#' @export
#' @describeIn settings_class Set distance units.
setReplaceMethod(
    f = "distance_units",
    signature = "settings",
    definition = function(x, value) {
  x@distance_units<- value
  return(x)
})





# #' @param x An object
# #'
# #' @describeIn settings_class Get the name of the time variable used in
# #'   formula
setMethod(
    f = "time_name",
    signature = "settings",
    definition = function(x) {
  return(time_name(formula(x)))
})
