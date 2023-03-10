% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generics.R, R/fit.R
\name{stass_simulate}
\alias{stass_simulate}
\alias{stass_simulate,stass-method}
\title{Simulate from a model object.}
\usage{
stass_simulate(object, ...)

\S4method{stass_simulate}{stass}(object, conditional = FALSE, ...)
}
\arguments{
\item{object}{A model object to simulate from.}

\item{...}{Extra options}

\item{conditional}{Logical. If false (default), new random effects and new
observations are simulated. If true, new observations are simulated
conditional on the random effect values in the supplied model.}
}
\value{
A single copy of \code{object} with a simulated dataset replacing
the original data
}
\description{
Simulate from a model object.
}
\section{Methods (by class)}{
\itemize{
\item \code{stass_simulate(stass)}: Simulate a new dataset from the model, with the
option to simulate a new set of random effects as well. The parameter
values used in the simulations are those set in \code{parameters(model)}.
Returns an \code{stass} object with simulated random effects (if
\code{conditional=FALSE})and simulated data.

}}
