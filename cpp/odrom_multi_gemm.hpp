#ifndef ODROM_MULTI_GEMM_HPP
#define ODROM_MULTI_GEMM_HPP

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"

#include "io.hpp"

template<class FomType>
struct OdRomMultiGemm
{

  // required
  using scalar_type   = double;
  using state_type    = Kokkos::View<double**>;
  using velocity_type = state_type;

  const std::vector<FomType> & fomObjs_;
  int numFoms_ = {};

  int numTiles_ = 0;
  Kokkos::View<int*> modesPerTile_;
  int totalModes_ = {};
  Kokkos::View<double***> phis_;
  Kokkos::View<int*> phiNumRowsPerTile_;
  Kokkos::View<double***> projs_;
  Kokkos::View<int*> projNumRowsPerTile_;

  mutable Eigen::MatrixXd fomStatesForPdaEvaluation_;
  Kokkos::LayoutStride fomStateLay_;
  Kokkos::View<double**, Kokkos::LayoutStride> fomStateView_;

  mutable Eigen::MatrixXd fomVelocitiesForPdaEvaluation_;
  Kokkos::LayoutStride fomVelocLay_;
  Kokkos::View<double**, Kokkos::LayoutStride> fomVelocView_;

  Kokkos::View<int*> romStateSpanStarts_;
  Kokkos::View<int*> fomStateSpanStarts_;
  Kokkos::View<int*> fomVelocSpanStarts_;

  Kokkos::View<double**> refStates_;

  OdRomMultiGemm(const std::vector<FomType> & fomObjs,
		 const YAML::Node & node)
    : fomObjs_(fomObjs),
      numFoms_(fomObjs_.size()),
      numTiles_(node["numTiles"].as<int>()),
      modesPerTile_("modesPerTile", numTiles_),
      phis_("phi", numTiles_, 1, 1),
      phiNumRowsPerTile_("phiNumRowsPerTile_", numTiles_),
      projs_("projs", numTiles_, 1, 1),
      projNumRowsPerTile_("projNumRowsPerTile_", numTiles_),
      //
      fomStatesForPdaEvaluation_(fomObjs[0].totalDofStencilMesh(), numFoms_),
      fomStateLay_(fomStatesForPdaEvaluation_.rows(), 1,
		   numFoms_, fomStatesForPdaEvaluation_.rows()),
      fomStateView_(fomStatesForPdaEvaluation_.data(), fomStateLay_),
      //
      fomVelocitiesForPdaEvaluation_(fomObjs[0].totalDofSampleMesh(), numFoms_),
      fomVelocLay_(fomVelocitiesForPdaEvaluation_.rows(), 1,
		   numFoms_, fomVelocitiesForPdaEvaluation_.rows()),
      fomVelocView_(fomVelocitiesForPdaEvaluation_.data(), fomVelocLay_),
      //
      romStateSpanStarts_("romStateSpanStarts", numTiles_),
      fomStateSpanStarts_("fomStateSpanStarts", numTiles_),
      fomVelocSpanStarts_("fomVelocSpanStarts", numTiles_),
      refStates_("refStates", fomObjs[0].totalDofStencilMesh(), numFoms_)
  {
    fomStatesForPdaEvaluation_.setZero();
    fomVelocitiesForPdaEvaluation_.setZero();

    for (int k=0; k<numFoms_; ++k){
      const auto fomIc = fomObjs[k].initialCondition();
      for (int i=0; i<fomIc.size(); ++i){
	refStates_(i,k) = fomIc(i);
      }
    }
    // write_rank1_view_to_ascii_file("ref_state.txt",
    // 				   Kokkos::subview(refStates_, Kokkos::ALL(), 0));

    //std::cout << "reading modesPerTile \n";
    read_integers_from_ascii_into_view("./modes_per_tile.txt", modesPerTile_);
    // for (int i=0; i<numTiles_; ++i){
    //   std::cout << "tileId = " << i << " modes = " << modesPerTile_(i) << '\n';
    // }
    // std::cout << "\n";

    // compute total num of modes
    Kokkos::parallel_reduce("sumModes", numTiles_,
			    KOKKOS_LAMBDA (const int i, int& update) {
			      update += modesPerTile_(i);
			    }, totalModes_);
    std::cout << "totalModes = " << totalModes_ << '\n';
    // std::cout << "\n";

    // find what is the max K over all tiles
    int maxK_ = 0;
    Kokkos::parallel_reduce("maxK", numTiles_,
			     KOKKOS_LAMBDA (const int& i, int& maxval) {
			       const int K = modesPerTile_(i);
			       if(K > maxval) maxval = K;
			     }, Kokkos::Max<int>(maxK_));
    // std::cout << "maxK_ = " << maxK_ << '\n';
    // std::cout << "\n";

    // read phi on stencil mesh for each tile
    const auto phiOnStencilMeshDir = node["phiOnStencilDir"].as<std::string>();
    read_integers_from_ascii_into_view(phiOnStencilMeshDir + "/rows_per_tile.txt", phiNumRowsPerTile_);
    const int maxRowsPhiOnStencil = read_single_int_from_ascii(phiOnStencilMeshDir+"/max_num_rows.txt");
    //std::cout << "maxRowsPhiOnStencil = " << maxRowsPhiOnStencil << '\n';
    Kokkos::resize(phis_, numTiles_, maxRowsPhiOnStencil, maxK_);
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
    const int maxRowsProjectors = read_single_int_from_ascii(projectorsDir+"/max_num_rows.txt");
    //std::cout << "maxRowsProjectors = " << maxRowsProjectors << '\n';
    Kokkos::resize(projs_, numTiles_, maxRowsProjectors, maxK_);
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

  }

  state_type createRomState() const {
    state_type res("crs", totalModes_, numFoms_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  velocity_type createVelocity() const {
    state_type res("cv", totalModes_, numFoms_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  void velocity(const state_type & romState,
		scalar_type evaltime,
		velocity_type & romRhs) const
  {
    doReconstructionTeam(romState);
    KokkosBlas::axpy(1., refStates_, fomStateView_);

    // auto fomState = fomStatesForPdaEvaluation_.col(0);
    // auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(0);
    // fomObjs_[0].velocity(fomState, evaltime, fomVeloc);

    // Kokkos::parallel_for(1,
    // 			   KOKKOS_LAMBDA(const int i)
    // 			   {
    // 			     auto fomState = fomStatesForPdaEvaluation_.col(i);
    // 			     auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(i);
    // 			     fomObjs_[i].velocity(fomState, evaltime, fomVeloc);
    // 			   });
    // Kokkos::fence();
// #pragma omp parallel for num_threads(16) //numFoms_)
//     for (int i=0; i<numFoms_; ++i){
//       auto fomState = fomStatesForPdaEvaluation_.col(i);
//       auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(i);
//       fomObjs_[i].velocity(fomState, evaltime, fomVeloc);
//     }

//#pragma omp barrier
      doProjectionTeam(romRhs);
    //Kokkos::fence();
  }

private:

  void doReconstructionTeam(const state_type & romState) const
  {

    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(numTiles_, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type member)
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
							     std::make_pair(rs_b, rs_e),
							     Kokkos::ALL());

			   const int fs_b = fomStateSpanStarts_(tileId);
			   const int fs_e = fs_b + myPhi_s0;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fs_b, fs_e),
							     Kokkos::ALL());

			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));
			   assert(myFomState.extent(1) == myRomState.extent(1));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemm::Unblocked;
			   KokkosBatched::TeamVectorGemm<
			     member_type, notr, notr, alg>::invoke(member,
								   1.0, myPhi,
								   myRomState,
								   0.0,
								   myFomState);
			 });
  }

  void doProjectionTeam(velocity_type & romRhs) const
  {
    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(numTiles_, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type member)
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
							   std::make_pair(rs_b, rs_e),
							   Kokkos::ALL());

			   const int fv_b = fomVelocSpanStarts_(tileId);
			   const int fv_e = fv_b + myProj_s0;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fv_b, fv_e),
							     Kokkos::ALL());

			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using tr = KokkosBatched::Trans::Transpose;
			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemm::Unblocked;
			   KokkosBatched::TeamVectorGemm<
			     member_type, tr, notr, alg>::invoke(member,
								 1.0, myProj,
								 myFomVeloc,
								 0.0,
								 myRomRhs);
			 });
  }

  void doReconstructionA(const state_type & romState) const
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
							     std::make_pair(rs_b, rs_e),
							     Kokkos::ALL());

			   const int fs_b = fomStateSpanStarts_(tileId);
			   const int fs_e = fs_b + myPhi_s0;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fs_b, fs_e),
							     Kokkos::ALL());

			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));
			   assert(myFomState.extent(1) == myRomState.extent(1));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemm::Blocked;
			   KokkosBatched::SerialGemm<notr, notr, alg>::invoke(1.0, myPhi,
									      myRomState,
									      0.0,
									      myFomState);
			 });
  }

  void doProjection(velocity_type & romRhs) const
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
							   std::make_pair(rs_b, rs_e),
							   Kokkos::ALL());

			   const int fv_b = fomVelocSpanStarts_(tileId);
			   const int fv_e = fv_b + myProj_s0;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fv_b, fv_e),
							     Kokkos::ALL());

			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using tr = KokkosBatched::Trans::Transpose;
			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemm::Blocked;
			   KokkosBatched::SerialGemm<tr, notr, alg>::invoke(1.0, myProj,
									    myFomVeloc,
									    0.0,
									    myRomRhs);
			 });
  }

};

#endif
