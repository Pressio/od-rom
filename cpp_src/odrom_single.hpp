#ifndef ODROM_SINGLE_HPP
#define ODROM_SINGLE_HPP

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"

#include "io.hpp"

template<class FomType>
struct OdRomSingle
{

  // required
  using scalar_type   = double;
  using state_type    = Kokkos::View<double*>;
  using velocity_type = state_type;

  const FomType & fomObj_;
  const int numTiles_ = 0;

  Kokkos::View<int*> modesPerTile_;
  int totalModes_ = {};
  int maxK_ = {};
  int minK_ = {};
  int avgK_ = {};

  Kokkos::View<double***> phis_;
  Kokkos::View<int*> phiNumRowsPerTile_;
  int maxRowsPhiOnStencil_ = {};
  Kokkos::View<double***> projs_;
  Kokkos::View<int*> projNumRowsPerTile_;
  int maxRowsProjectors_ = {};

  mutable Eigen::VectorXd fomStatesForPdaEvaluation_;
  Kokkos::LayoutStride fomStateLay_;
  Kokkos::View<double*, Kokkos::LayoutStride> fomStateView_;

  mutable Eigen::VectorXd fomVelocitiesForPdaEvaluation_;
  Kokkos::LayoutStride fomVelocLay_;
  Kokkos::View<double*, Kokkos::LayoutStride> fomVelocView_;

  Kokkos::View<int*> romStateSpanStarts_;
  Kokkos::View<int*> fomStateSpanStarts_;
  Kokkos::View<int*> fomVelocSpanStarts_;

  Kokkos::View<double*> refStates_;

  OdRomSingle(const FomType & fomObj, const YAML::Node & node)
    : fomObj_(fomObj),
      numTiles_(node["numTiles"].as<int>()),
      modesPerTile_("modesPerTile", numTiles_),
      phis_("phi", numTiles_, 1, 1),
      phiNumRowsPerTile_("phiNumRowsPerTile_", numTiles_),
      projs_("projs", numTiles_, 1, 1),
      projNumRowsPerTile_("projNumRowsPerTile_", numTiles_),
      //
      fomStatesForPdaEvaluation_(fomObj.totalDofStencilMesh()),
      fomStateLay_(fomStatesForPdaEvaluation_.rows(), 1),
      fomStateView_(fomStatesForPdaEvaluation_.data(), fomStateLay_),
      //
      fomVelocitiesForPdaEvaluation_(fomObj.totalDofSampleMesh()),
      fomVelocLay_(fomVelocitiesForPdaEvaluation_.rows(), 1),
      fomVelocView_(fomVelocitiesForPdaEvaluation_.data(), fomVelocLay_),
      //
      romStateSpanStarts_("romStateSpanStarts", numTiles_),
      fomStateSpanStarts_("fomStateSpanStarts", numTiles_),
      fomVelocSpanStarts_("fomVelocSpanStarts", numTiles_),
      refStates_("refStates", fomObj.totalDofStencilMesh())
  {
    fomStatesForPdaEvaluation_.setZero();
    fomVelocitiesForPdaEvaluation_.setZero();

    const auto fomIc = fomObj.initialCondition();
    for (int i=0; i<fomIc.size(); ++i){
      refStates_(i) = fomIc(i);
    }

    //std::cout << "reading modesPerTile \n";
    read_integers_from_ascii_into_view("./modes_per_tile.txt", modesPerTile_);
    double avg = {};
    for (int i=0; i<modesPerTile_.extent(0); ++i){
      avg += (double) modesPerTile_(i);
    }
    avgK_ = int(avg/(double) numTiles_);

    // for (int i=0; i<numTiles_; ++i){
    //   std::cout << "tileId = " << i << " modes = " << modesPerTile_(i) << '\n';
    // }
    // std::cout << "\n";

    // compute total num of modes
    Kokkos::parallel_reduce("sumModes", numTiles_,
			    KOKKOS_LAMBDA (const int i, int& update) {
			      update += modesPerTile_(i);
			    }, totalModes_);

    // find what is the min/max K over all tiles
    Kokkos::parallel_reduce("maxK", numTiles_,
			     KOKKOS_LAMBDA (const int& i, int& maxval) {
			       const int K = modesPerTile_(i);
			       if(K > maxval) maxval = K;
			     }, Kokkos::Max<int>(maxK_));
    Kokkos::parallel_reduce("minK", numTiles_,
			     KOKKOS_LAMBDA (const int& i, int& val) {
			       const int K = modesPerTile_(i);
			       if(K < val) val = K;
			     }, Kokkos::Min<int>(minK_));
    // std::cout << "\n";

    // read phi on stencil mesh for each tile
    const auto phiOnStencilMeshDir = node["phiOnStencilDir"].as<std::string>();
    read_integers_from_ascii_into_view(phiOnStencilMeshDir + "/rows_per_tile.txt", phiNumRowsPerTile_);
    maxRowsPhiOnStencil_ = read_single_int_from_ascii(phiOnStencilMeshDir+"/max_num_rows.txt");
    Kokkos::resize(phis_, numTiles_, maxRowsPhiOnStencil_, maxK_);
    for (int i=0; i<numTiles_; ++i)
    {
      const auto myK = modesPerTile_(i);
      auto sv = Kokkos::subview(phis_, i, Kokkos::ALL(), std::make_pair(0, myK));
      const auto currFile = phiOnStencilMeshDir + "/phi_on_stencil_p_" + std::to_string(i) + ".txt";
      read_phi_on_stencil_from_ascii_file(currFile, sv);
    }
    //std::cout << "\n";

    // read projectors for each tile
    const auto projectorsDir = node["projectorDir"].as<std::string>();
    read_integers_from_ascii_into_view(projectorsDir + "/rows_per_tile.txt", projNumRowsPerTile_);
    maxRowsProjectors_ = read_single_int_from_ascii(projectorsDir+"/max_num_rows.txt");
    Kokkos::resize(projs_, numTiles_, maxRowsProjectors_, maxK_);
    for (int i=0; i<numTiles_; ++i)
    {
      const auto myK = modesPerTile_(i);
      auto sv = Kokkos::subview(projs_, i, Kokkos::ALL(), std::make_pair(0, myK));
      const auto currFile = projectorsDir + "/projector_p_" + std::to_string(i) + ".txt";
      read_projector_from_ascii_file(currFile, sv);
    }
    //std::cout << "\n";

    // store bounds info needed to subviews
    Kokkos::parallel_scan(numTiles_,
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = modesPerTile_(i);
			    if (final) { romStateSpanStarts_(i) = update; }
			    update += val_i;
			  });
    Kokkos::parallel_scan(numTiles_,
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = phiNumRowsPerTile_(i);
			    if (final) { fomStateSpanStarts_(i) = update; }
			    update += val_i;
			  });
    Kokkos::parallel_scan(numTiles_,
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = projNumRowsPerTile_(i);
			    if (final) { fomVelocSpanStarts_(i) = update; }
			    update += val_i;
			  });

    // for (int i=0; i<numTiles_; ++i){
    //   std::cout << "romStateSS(i) = " << i << " " << romStateSpanStarts_(i)
    // 		<< " "
    // 		<< "fomStateSS(i) = " << i << " " << fomStateSpanStarts_(i)
    // 		<< " "
    // 		<< "fomVelocSS(i) = " << i << " " << fomVelocSpanStarts_(i)
    // 		<< '\n';
    // }

    printInfo();
  }

  void printInfo()
  {
    std::cout << "--------------------------------------------------\n";
    std::cout << "totNumTiles_                = " << numTiles_ << '\n';
    std::cout << "totModes                    = " << totalModes_ << '\n';
    std::cout << "{min,max}NumModesOverTiles  = " << minK_ << " " << maxK_ << '\n';
    std::cout << "avg_NumModesOverTiles       = " << avgK_ << '\n';
    std::cout << "maxRowsPhiOnStencil         = " << maxRowsPhiOnStencil_ << '\n';
    std::cout << "maxRowsProjectors           = " << maxRowsProjectors_ << '\n';
    std::cout << "\n";

    std::cout << "romStateSize = " << totalModes_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "romRhsSize   = " << totalModes_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";

    std::cout << "fomStateSize = " <<
      fomStatesForPdaEvaluation_.rows()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "fomRhsSize   = " <<
      fomVelocitiesForPdaEvaluation_.rows()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";

    std::cout << "--------------------------------------------------";
    std::cout << "\n";
  }

  state_type createRomState() const {
    state_type res("crs", totalModes_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  velocity_type createVelocity() const {
    state_type res("cv", totalModes_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  void velocity(const state_type & romState,
		scalar_type evaltime,
		velocity_type & romRhs) const
  {
    doReconstructionTeamPol(romState);
    KokkosBlas::axpy(1., refStates_, fomStateView_);

    //#pragma omp barrier
    fomObj_.velocity(fomStatesForPdaEvaluation_, evaltime, fomVelocitiesForPdaEvaluation_);

// #pragma omp barrier
    doProjectionTeamPol(romRhs);
//     Kokkos::fence();
  }

private:
  void doReconstructionTeamPol(const state_type & romState) const
  {
    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(numTiles_, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type & member)
			 {
			   const std::size_t tileId = member.league_rank();
			   const int myK = modesPerTile_[tileId];

			   const int myPhi_s0 = phiNumRowsPerTile_(tileId);
			   auto myPhi = Kokkos::subview(phis_, tileId,
							std::make_pair(0, myPhi_s0),
							std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomState = Kokkos::subview(romState,
							     std::make_pair(rs_b, rs_e));

			   const int fs_b = fomStateSpanStarts_(tileId);
			   const int fs_e = fs_b + myPhi_s0;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fs_b, fs_e));

			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			   KokkosBatched::TeamVectorGemv<
			     member_type, notr, alg>::invoke(member, 1.0, myPhi, myRomState,
							     0.0, myFomState);
			 });

  }

  void doProjectionTeamPol(velocity_type & romRhs) const
  {
    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(numTiles_, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type & member)
			 {
			   const std::size_t tileId = member.league_rank();
			   const int myK = modesPerTile_[tileId];

			   const int myProj_s0 = projNumRowsPerTile_(tileId);
			   auto myProj = Kokkos::subview(projs_, tileId,
							 std::make_pair(0, myProj_s0),
							 std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomRhs = Kokkos::subview(romRhs,
							   std::make_pair(rs_b, rs_e));

			   const int fv_b = fomVelocSpanStarts_(tileId);
			   const int fv_e = fv_b + myProj_s0;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fv_b, fv_e));

			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using tr = KokkosBatched::Trans::Transpose;
			   using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			   KokkosBatched::TeamVectorGemv<
			     member_type, tr, alg>::invoke(member, 1.0, myProj, myFomVeloc,
							   0.0, myRomRhs);
			 });
  }

  void doReconstructionRangePol(const state_type & romState) const
  {
    Kokkos::parallel_for(numTiles_,
			 KOKKOS_LAMBDA(const std::size_t tileId)
			 {
			   const int myK = modesPerTile_[tileId];

			   const int myPhi_s0 = phiNumRowsPerTile_(tileId);
			   auto myPhi = Kokkos::subview(phis_, tileId,
							std::make_pair(0, myPhi_s0),
							std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomState = Kokkos::subview(romState,
							     std::make_pair(rs_b, rs_e));

			   const int fs_b = fomStateSpanStarts_(tileId);
			   const int fs_e = fs_b + myPhi_s0;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fs_b, fs_e));

			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemv::Blocked;
			   KokkosBatched::SerialGemv<
			     notr, alg>::invoke(1.0, myPhi, myRomState, 0.0, myFomState);
			 });
  }

  void doProjectionRangePol(velocity_type & romRhs) const
  {

    Kokkos::parallel_for(numTiles_,
			 KOKKOS_LAMBDA(const std::size_t tileId)
			 {
			   const int myK = modesPerTile_[tileId];

			   const int myProj_s0 = projNumRowsPerTile_(tileId);
			   auto myProj = Kokkos::subview(projs_, tileId,
							 std::make_pair(0, myProj_s0),
							 std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomRhs = Kokkos::subview(romRhs,
							   std::make_pair(rs_b, rs_e));

			   const int fv_b = fomVelocSpanStarts_(tileId);
			   const int fv_e = fv_b + myProj_s0;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fv_b, fv_e));

			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using tr = KokkosBatched::Trans::Transpose;
			   using alg  = KokkosBatched::Algo::Gemv::Blocked;
			   KokkosBatched::SerialGemv<
			     tr, alg>::invoke(1.0, myProj, myFomVeloc, 0.0, myRomRhs);
			 });
  }


};

#endif
