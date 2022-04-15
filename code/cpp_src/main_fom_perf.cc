
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "pressiodemoapps/advection_diffusion2d.hpp"
#include "pressiodemoapps/swe2d.hpp"
#include <chrono>
#include "CLI11.hpp"

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;

template<class T = void>
pode::StepScheme string_to_ode_scheme(const std::string & schemeString)
{

  if (schemeString == "ForwardEuler"){
    return pode::StepScheme::ForwardEuler;
  }
  else if (schemeString == "RungeKutta4"){
    return pode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "RK4"){
    return pode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "SSPRungeKutta3"){
    return pode::StepScheme::SSPRungeKutta3;
  }
  else if (schemeString == "SSPRK3"){
    return pode::StepScheme::SSPRungeKutta3;
  }
  else{
    throw std::runtime_error("string_to_ode_scheme: Invalid odeScheme");
  }

}

pda::InviscidFluxReconstruction
stringToInviscidRecScheme(const std::string & string)
{
  if (string == "FirstOrder"){
    return pda::InviscidFluxReconstruction::FirstOrder;
  }
  else if (string == "Weno3"){
    return pda::InviscidFluxReconstruction::Weno3;
  }
  else if (string == "Weno5"){
    return pda::InviscidFluxReconstruction::Weno5;
  }

  return {};
}

int main(int argc, char *argv[])
{
  CLI::App app;
  std::string meshDir = "void";
  int nth   = 0;
  int loopCount = 100;
  app.add_option("-m", meshDir);
  app.add_option("-l", loopCount);
  app.add_option("-n", nth);
  CLI11_PARSE(app, argc, argv);

  const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(meshDir);

  // const auto probE = pda::AdvectionDiffusion2d::Burgers;
  // const auto viscSchemeE = pda::ViscousFluxReconstruction::FirstOrder;
  const auto inviscidSchemeE = stringToInviscidRecScheme("Weno5");
  // auto appObj = pda::create_problem_eigen(meshObj, probE, inviscidSchemeE, viscSchemeE);
  auto appObj = pda::create_problem_eigen(meshObj, pda::Swe2d::SlipWall, inviscidSchemeE);
  //appObj.setNumThreads(nth);

  // check velocity evaluation
  {
    auto state = appObj.initialCondition();
    auto V = appObj.createVelocity();
    std::cout << "fomStateSize = " << state.size()*sizeof(double)/(double) 1e6 << " (MB)\n";
    std::cout << "fomRhsSize   = " << V.size()*sizeof(double)/(double) 1e6 << " (MB)\n";

    // warmup
    appObj.velocity(state, 0., V);

//     // velo + update only
//     {
//       auto t1 = std::chrono::high_resolution_clock::now();
//       for (int i=0; i<loopCount; ++i){
// 	appObj.velocity(state, 0., V);

// 	{
// #pragma omp parallel for schedule(static)
// 	  for (int k=0; k<state.size(); ++k){
// 	    state(k) += 0.4*V(k);
// 	  }
// 	}
//       }
//       auto t2 = std::chrono::high_resolution_clock::now();
//       std::chrono::duration< double > fs = t2 - t1;
//       std::cout << "singleVeloAndUpdateEval " << fs.count()/(double) loopCount << std::endl;
//     }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<loopCount; ++i){
      appObj.velocity(state, 0., V);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > fs = t2 - t1;
    std::cout << "singleVeloEval " << fs.count()/(double) loopCount << std::endl;


  }

  // // check real time stepping
  // // this is not the same as just velo evaluation because it also involves
  // // state updating so that has some effect
  // {
  //   const auto odeSchemeE = string_to_ode_scheme("RK4");
  //   auto stepperObj = pode::create_explicit_stepper(odeSchemeE, state, appObj);
  //   auto t1 = std::chrono::high_resolution_clock::now();
  //   pode::advance_n_steps(stepperObj, state, 0., 0.0001, numSteps);
  //   auto t2 = std::chrono::high_resolution_clock::now();
  //   std::chrono::duration< double > fs = t2 - t1;
  //   std::cout << "singleTimeStepRK4 " << fs.count()/(double) numSteps << std::endl;
  // }

  return 0;
}
