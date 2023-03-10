% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classes.R, R/settings.R
\docType{class}
\name{settings-class}
\alias{settings-class}
\alias{formula,settings-method}
\alias{formula<-,settings-method}
\alias{distance_units,settings-method}
\alias{distance_units<-,settings-method}
\title{An S4 class to hold settings for a stass model}
\usage{
\S4method{formula}{settings}(x)

\S4method{formula}{settings}(x) <- value

\S4method{distance_units}{settings}(x)

\S4method{distance_units}{settings}(x) <- value
}
\arguments{
\item{x}{An object}

\item{value}{A replacement value}
}
\description{
An S4 class to hold settings for a stass model
}
\section{Functions}{
\itemize{
\item \code{formula(settings)}: Get model formula

\item \code{formula(settings) <- value}: Set model formula

\item \code{distance_units(settings)}: Get distance units used for the model.

\item \code{distance_units(settings) <- value}: Set distance units.

}}
\section{Slots}{

\describe{
\item{\code{formula}}{A formula}

\item{\code{distance_units}}{The units used for distance calculation.  Must be
compatible with units::set_units.}
}}

\seealso{
Other stass_classes: 
\code{\link{TMB_out-class}},
\code{\link{dag-class}},
\code{\link{long_stars-class}},
\code{\link{observation_parameters-class}},
\code{\link{observations-class}},
\code{\link{parameters-class}},
\code{\link{process-class}},
\code{\link{process_parameters-class}},
\code{\link{stass-class}},
\code{\link{tracing-class}}
}
\concept{stass_classes}
\keyword{internal}
