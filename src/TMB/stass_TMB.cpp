#define TMB_LIB_INIT R_init_stass
#include <TMB.hpp>
using namespace density;

#include "include/utils.hpp"
#include "include/covariance.hpp"
#include "include/conditional_normal.hpp"
#include "include/time_series.hpp"
#include "include/dag.hpp"
#include "include/persistent_graph.hpp"
#include "include/pg_cache.hpp"
#include "include/transient_graph.hpp"
#include "include/tg_cache.hpp"
#include "include/nngp.hpp"
#include "include/observations.hpp"

#include "model/model.hpp"

template<class Type>
Type objective_function<Type>::operator() () {
  DATA_STRING(model);
  if( model == "model" ) {
    return stass_model(this);
  } else {
    error("Unknown model.");
  }
  return 0;
}
