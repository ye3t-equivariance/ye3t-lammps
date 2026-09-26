/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#ifndef LMP_YE3T_KOKKOS_STEP_STATE_H
#define LMP_YE3T_KOKKOS_STEP_STATE_H

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>

namespace YE3T_LAMMPS {

struct KokkosEdgeSummary {
  std::int64_t edge_count = 0;
  int maximum_edges = 0;
  int status = 0;
};

// Exclusive prefix sum over the per-center edge counts, with a fused final
// publication of allocation/status metadata. The count/max kernels finish on
// this default execution instance before the scan begins.
template <class DeviceType> struct KokkosScanEdgesSummary {
  Kokkos::View<const std::int64_t *, DeviceType> counts;
  Kokkos::View<std::int64_t *, DeviceType> offsets;
  int center_count = 0;
  Kokkos::View<KokkosEdgeSummary, DeviceType> summary;
  Kokkos::View<const int *, DeviceType> status;
  Kokkos::View<const int, DeviceType> maximum;
  bool include_maximum = false;
  using value_type = std::int64_t;
  KOKKOS_INLINE_FUNCTION
  void operator()(int index, std::int64_t &update, bool final) const
  {
    if (final) {
      offsets(index) = update;
      if (index == center_count) {
        summary().edge_count = update;
        summary().maximum_edges = include_maximum ? maximum() : 0;
        summary().status = status(0);
      }
    }
    if (index < center_count) update += counts(index);
  }
};

// Small persistent device results, not atom/neighbor caches. All operations use
// the SAME default execution-space instance as the pair kernels. No separate
// stream or cross-stream event assumption is introduced.
template <class DeviceType, class Tally> class KokkosStepState {
 public:
  struct Totals {
    Tally ev{};
    double maximum_imaginary = 0.0;
  };
  using IntScalar = Kokkos::View<int, DeviceType>;
  using DoubleScalar = Kokkos::View<double, DeviceType>;
  using TallyScalar = Kokkos::View<Tally, DeviceType>;
  using SummaryScalar = Kokkos::View<KokkosEdgeSummary, DeviceType>;
  using TotalsScalar = Kokkos::View<Totals, DeviceType>;

  IntScalar maximum_edges;
  DoubleScalar imaginary_chunk;
  TallyScalar ev_chunk;

  void reset()
  {
    if (!totals_.data()) {
      maximum_edges = IntScalar("ye3t:maximum_edges");
      imaginary_chunk = DoubleScalar("ye3t:imaginary_chunk");
      ev_chunk = TallyScalar("ye3t:ev_chunk");
      edge_summary_ = SummaryScalar("ye3t:edge_summary");
      totals_ = TotalsScalar("ye3t:step_totals");
    }
    Kokkos::deep_copy(DeviceType{}, totals_, Totals{});
  }

  SummaryScalar summary_view() const { return edge_summary_; }

  KokkosEdgeSummary read_edge_summary() const
  {
    // The existing prefix scan publishes the summary in its final iteration;
    // packing requires no extra kernel launch.
    const DeviceType exec{};
    KokkosEdgeSummary host;
    Kokkos::deep_copy(exec, host, edge_summary_);
    exec.fence("YE3T edge allocation/status decision");
    return host;
  }

  void accumulate(bool include_ev, bool include_imaginary) const
  {
    if (!include_ev && !include_imaginary) return;
    const auto totals = totals_;
    const auto ev = ev_chunk;
    const auto imaginary = imaginary_chunk;
    Kokkos::parallel_for(
        "YE3TAccumulateChunkTallies", Kokkos::RangePolicy<DeviceType>(0, 1),
        KOKKOS_LAMBDA(const int) {
          if (include_ev) totals().ev += ev();
          if (include_imaginary && imaginary() > totals().maximum_imaginary)
            totals().maximum_imaginary = imaginary();
        });
  }

  Totals read_totals() const
  {
    const DeviceType exec{};
    Totals host;
    Kokkos::deep_copy(exec, host, totals_);
    exec.fence("YE3T timestep tallies");
    return host;
  }

  // Same single fence, additionally collecting the accumulated device status
  // word so a timestep needs no separate blocking status read.
  template <class StatusView> Totals read_totals(const StatusView &status, int &status_value) const
  {
    const DeviceType exec{};
    Totals host;
    Kokkos::deep_copy(exec, host, totals_);
    Kokkos::deep_copy(exec, status_value, Kokkos::subview(status, 0));
    exec.fence("YE3T timestep tallies and deferred status");
    return host;
  }

  std::size_t memory_usage() const
  {
    return totals_.data()
        ? sizeof(int) + sizeof(double) + sizeof(Tally) + sizeof(KokkosEdgeSummary) + sizeof(Totals)
        : 0;
  }

 private:
  SummaryScalar edge_summary_;
  TotalsScalar totals_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_KOKKOS_STEP_STATE_H
