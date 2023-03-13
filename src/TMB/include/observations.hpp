template<class Type>
class observations {
  // private:
  public:
    vector<Type> removals; // [idx, l]
    vector<Type> effort; // [idx, age]
    array<int> idx; // 2d array, each row gives [location, time]
    array<Type> response_pars; // [par, var
  // public:
    observations(
      const vector<Type>& removals,
      const vector<Type>& effort,
      const array<int>& idx,
      const array<Type>& response_pars
    ) :
    removals{removals},
    effort{effort},
    idx{idx},
    response_pars{response_pars} {};

    Type loglikelihood(nngp<Type>& process) {
      Type ans = 0.0;
      for(int i = 0; i < removals.size(); i++) {
        Type pred_removals = response_pars(0, 0) * 
          exp(process(idx(i, 0), idx(i, 1), 1)) *
          process(idx(i, 0), idx(i, 1), 0);
        // Log-normal distribution
        ans += dnorm(
          log( removals(i) / pred_removals),
          Type(0.0),
          response_pars(1, 0),
          true
        );
        ans -= log( removals(i) / pred_removals);

        ans += dnorm(
          effort(i),
          exp(process(idx(i, 0), idx(i, 1), 1)),
          response_pars(0, 1),
          true
        );
      }
      return ans;
    }

    // observations<Type> simulate(nngp<Type>& process) {
    //   array<Type> growth_parameters = get_growth_parameters(process);
    //   for(int i = 0; i < l.size(); i++) {
    //     Type pred_l = growth_curve(
    //       age(i),
    //       growth_parameters(i, 0),
    //       growth_parameters(i, 1)
    //     );
    //     l(i) = pred_l * exp(rnorm(Type(0.0), sd));
    //   }
    //   return *this;
    // }
};