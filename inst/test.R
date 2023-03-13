library(stass)
bird_survey<- starve::bird_survey
bird_survey$e<- 1
mesh<- INLA::inla.mesh.2d(
  sf::st_coordinates(bird_survey),
  max.edge = 2,
  cutoff = 1.5,
  crs = as(sf::st_crs(bird_survey), "CRS")
)
foo<- stass_prepare(
  cnt ~ effort(e) + time(year),
  bird_survey,
  mesh
)

TMB_input<- stass:::convert_to_TMB_list(foo)

obj<- TMB::MakeADFun(
  data = TMB_input$data,
  para = TMB_input$para,
  random = TMB_input$rand,
  map = TMB_input$map,
  DLL = "stass_TMB",
  silent = FALSE
)
