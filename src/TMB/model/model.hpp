#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

template<class Type>
Type stass_model(objective_function<Type>* obj) {
  DATA_INTEGER(conditional_sim); // If true, use old values of time series and process random effects
  Type nll = 0.0;

  /*
  Start of temporal component
  */
  PARAMETER_ARRAY(ts_re); // [time,var]
  PARAMETER_ARRAY(working_ts_pars); // [par,var]

  array<Type> ts_pars = working_ts_pars;
  for(int v = 0; v < ts_pars.dim(1); v++) {
    ts_pars(0, v) = working_ts_pars(0, v); // mean, no transformation
    ts_pars(1, v) = 2 * invlogit(working_ts_pars(1, v)) - 1; // ar1, --> (-1,1)
    ts_pars(2, v) = exp(working_ts_pars(2, v)); // marginal sd, --> (0,Inf)
  }
  REPORT(ts_pars);
  ADREPORT(ts_pars);

  time_series<Type> ts {ts_re, ts_pars};
  nll -= ts.loglikelihood();
  SIMULATE{
    if( !conditional_sim ) {
      ts_re = ts.simulate().get_re();
    }
    REPORT(ts_re);
  }
  /*
  End of temporal component
  */


  /*
  Start of spatio-temporal component
  */
  //  Set up persistent graph
  PARAMETER_ARRAY(pg_re); // [space,time,var]
  DATA_STRUCT(pg_edges, directed_graph);
  DATA_STRUCT(pg_dists, dag_dists);
  dag<Type> pg_g {pg_edges.dag, pg_dists.dag_dist};
  persistent_graph<Type> pg {pg_re, pg_re, pg_g};

  // Set up transient graph
  PARAMETER_ARRAY(tg_re); // [idx,var]
  DATA_IVECTOR(tg_t);
  DATA_STRUCT(tg_edges, directed_graph);
  DATA_STRUCT(tg_dists, dag_dists);
  dag<Type> tg_g {tg_edges.dag, tg_dists.dag_dist};
  transient_graph<Type> tg {tg_re, tg_re, tg_g, tg_t, pg.dim_t()};

  // Set up covariance functions
  DATA_IVECTOR(cv_code);
  PARAMETER_ARRAY(working_cv_pars); // [par,var], columns may have trailing NA
  array<Type> cv_pars = working_cv_pars;
  for(int v = 0; v < cv_code.size(); v++) {
    switch(cv_code(v)) {
      case 0 :
        cv_pars(0, v) = exp(working_cv_pars(0, v));
        cv_pars(1, v) = exp(working_cv_pars(1, v));
        cv_pars(2, v) = exp(working_cv_pars(2, v));
        break; // Exponential [sd,range] --> [(0,Inf), (0,Inf)]
      case 1 :
        cv_pars(0, v) = exp(working_cv_pars(0, v));
        cv_pars(1, v) = exp(working_cv_pars(1, v));
        cv_pars(2, v) = exp(working_cv_pars(2, v));
        break; // Gaussian [marg. sd, range] --> [(0,Inf), (0,Inf)]
      case 2 :
        cv_pars(0, v) = exp(working_cv_pars(0, v));
        cv_pars(1, v) = exp(working_cv_pars(1, v));
        cv_pars(2, v) = exp(working_cv_pars(2, v));
        break; // Matern [sd, range, nu] --> [(0,Inf), (0,Inf), (0,Inf)]
      case 3 :
        cv_pars(0, v) = exp(working_cv_pars(0, v));
        cv_pars(1, v) = exp(working_cv_pars(1, v));
        cv_pars(2, v) = exp(working_cv_pars(2, v));
        break; // Matern32 [sd, range] --> [(0,Inf), (0,Inf)]
      default :
        cv_pars(0, v) = exp(working_cv_pars(0, v));
        cv_pars(1, v) = exp(working_cv_pars(1, v));
        cv_pars(2, v) = exp(working_cv_pars(2, v));
        break; // Exponential [sd,range] --> [(0,Inf), (0,Inf)]
    }
  }
  REPORT(cv_pars);
  ADREPORT(cv_pars);
  vector<covariance<Type> > cv(cv_code.size());
  for(int v = 0; v < cv_code.size(); v++) {
    cv(v) = covariance<Type> {
      vector<Type>(cv_pars.col(v)),
      cv_code(v)
    };
  }

  // Spatio-temporal component
  nngp<Type> process {pg, tg, cv};
  nll -= process.loglikelihood(ts);
  SIMULATE{
    if( !conditional_sim ) {
      process.simulate(ts);
      pg_re = process.get_pg_re();
      tg_re = process.get_tg_re();
    }
    REPORT(pg_re);
    REPORT(tg_re);
  }
  /*
  End of spatio-temporal component
  */



  /*
  Start of observation component
  */
  DATA_ARRAY(length); // [idx, length]
  DATA_ARRAY(age); // [idx, age]
  DATA_IARRAY(idx); // [idx, (graph node, time)]
  DATA_MATRIX(mean_design); // [idx, covar]
  PARAMETER_ARRAY(beta); // [covar, var]
  PARAMETER_ARRAY(working_response_pars); // [par,var], columns may have trailing NA
  array<Type> response_pars = working_response_pars;
  response_pars(0, 0) = exp(working_response_pars(0, 0));
  REPORT(response_pars);
  ADREPORT(response_pars);

  vonBertalanffy<Type> vonB {length, age, idx, mean_design, beta, response_pars(0, 0)};
  nll -= vonB.loglikelihood(process);
  SIMULATE{
    vonB.simulate(process);
    length = vonB.l;
    REPORT(length);
  }
  /*
  End of observation component
  */

  /*
  Start of prediction component
  */
  DATA_STRUCT(pred_edges, directed_graph);
  DATA_STRUCT(pred_dists, dag_dists);
  dag<Type> pred_g {pred_edges.dag, pred_dists.dag_dist};
  DATA_IVECTOR(pred_t);
  PARAMETER_ARRAY(pred_re);

  nll -= process.prediction_loglikelihood(pred_g, pred_t, pred_re, ts);
  /*
  End of prediction component
  */

  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this