template<class Type>
class vonBertalanffy {
  // private:
  public:
    vector<Type> l; // [idx, l]
    vector<Type> age; // [idx, age]
    array<int> idx; // 2d array, each row gives [location, time]
    matrix<Type> mean_design; // [obs_idx, covariate]
    matrix<Type> beta; // [par, var]
    Type sd;
  // public:
    vonBertalanffy(
      const vector<Type>& l,
      const vector<Type>& age,
      const array<int>& idx,
      const matrix<Type>& mean_design,
      const matrix<Type>& beta,
      const Type sd
    ) :
    l{l},
    age{age},
    idx{idx},
    mean_design{mean_design},
    beta{beta},
    sd{sd} {};

    array<Type> get_growth_parameters(nngp<Type>& process) {
      array<Type> m(l.size(), 2);
      for(int i = 0; i < m.dim(0); i++) {
        for(int v = 0; v < m.dim(1); v++) {
          m(i, v) = vector<Type>(mean_design.row(i) * beta.col(v)).sum() +
            process(idx(i, 0), idx(i, 1), v);
          // m(i, v) = exp(m(i, v));
        }
      }
      return m;
    }

    Type growth_curve(Type a, Type linf, Type k) {
      return linf * (1 - exp(-k * a));
    }

    Type loglikelihood(nngp<Type>& process) {
      Type ans = 0.0;
      array<Type> growth_parameters = get_growth_parameters(process);
      for(int i = 0; i < l.size(); i++) {
        Type pred_l = growth_curve(
          age(i),
          growth_parameters(i, 0),
          growth_parameters(i, 1)
        );
        // Log-normal distribution
        ans += dnorm(
          log( l(i) / pred_l),
          Type(0.0),
          sd,
          true
        );
        ans -= log( l(i) / pred_l);
      }
      return ans;
    }

    vonBertalanffy<Type> simulate(nngp<Type>& process) {
      array<Type> growth_parameters = get_growth_parameters(process);
      for(int i = 0; i < l.size(); i++) {
        Type pred_l = growth_curve(
          age(i),
          growth_parameters(i, 0),
          growth_parameters(i, 1)
        );
        l(i) = pred_l * exp(rnorm(Type(0.0), sd));
      }
      return *this;
    }
};