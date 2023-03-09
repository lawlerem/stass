#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

template<class Type>
Type length_at_age(objective_function<Type>* obj) {
  PARAMETER(max);
  PARAMETER(rate);
  DATA_SCALAR(age);

  Type length = max * (1 - exp( -rate * age));
  return length;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this