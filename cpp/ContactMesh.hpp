// Copyright (C) 2022 Jørgen S. Dokken and Igor Baratta
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace dolfinx_contact
{
dolfinx::mesh::Mesh
update_ghosts(const dolfinx::mesh::Mesh& mesh,
              const dolfinx::graph::AdjacencyList<std::int32_t>& dest)
{
  // Get topology information
  const dolfinx::mesh::Topology& topology = mesh.topology();
  int tdim = topology.dim();

  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = topology.index_map(tdim);
  std::shared_ptr<const dolfinx::common::IndexMap> vert_map
      = topology.index_map(0);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> cv
      = topology.connectivity(tdim, 0);
  assert(cv);

  std::int32_t num_local_cells = cell_map->size_local();
  std::int32_t num_ghosts = cell_map->num_ghosts();

  // Get geometry information
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  int gdim = geometry.dim();
  xtl::span<const double> coord = geometry.x();

  std::vector<std::int32_t> vertex_to_coord(vert_map->size_local()
                                            + vert_map->num_ghosts());
  for (std::int32_t c = 0; c < num_local_cells + num_ghosts; ++c)
  {
    auto vertices = cv->links(c);
    auto dofs = geometry.dofmap().links(c);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      vertex_to_coord[vertices[i]] = dofs[i];
  }

  std::vector<std::int64_t> topology_array;
  std::vector<std::int32_t> counter(num_local_cells);
  std::vector<int64_t> global_inds(cv->num_links(0));

  // Compute topology information
  for (std::int32_t i = 0; i < num_local_cells; i++)
  {
    vert_map->local_to_global(cv->links(i), global_inds);
    topology_array.insert(topology_array.end(), global_inds.begin(),
                          global_inds.end());
    counter[i] += global_inds.size();
  }

  std::vector<std::int32_t> offsets(counter.size() + 1, 0);
  std::partial_sum(counter.begin(), counter.end(), offsets.begin() + 1);
  dolfinx::graph::AdjacencyList<std::int64_t> cell_vertices(topology_array,
                                                            offsets);

  // Copy over existing mesh vertices
  const std::int32_t num_local_vertices = vert_map->size_local();
  xt::xtensor<double, 2> x = xt::empty<double>({num_local_vertices, gdim});
  for (int v = 0; v < num_local_vertices; ++v)
    for (int j = 0; j < gdim; ++j)
      x(v, j) = coord[vertex_to_coord[v] * 3 + j];

  // Need to supply all input arguments to work with all compilers
  auto partitioner =
      [&dest](
          [[maybe_unused]] MPI_Comm comm, [[maybe_unused]] int nparts,
          [[maybe_unused]] int tdim,
          [[maybe_unused]] const dolfinx::graph::AdjacencyList<int64_t>& cells,
          [[maybe_unused]] dolfinx::mesh::GhostMode ghost_mode)
  { return dest; };
  dolfinx::mesh::Mesh new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), cell_vertices, geometry.cmap(), x,
      dolfinx::mesh::GhostMode::shared_facet, partitioner);
  return new_mesh;
}

} // namespace dolfinx_contact

namespace dolfinx_contact
{
// dolfinx::mesh::Mesh
dolfinx::graph::AdjacencyList<std::int32_t>
add_ghost_cells(const dolfinx::mesh::Mesh& mesh,
                const tcb::span<const std::int32_t>& cells)
{
  std::uint8_t cell_indicator = cells.size() > 0 ? 1 : 0;
  MPI_Comm comm = mesh.comm();
  const int mpi_rank = dolfinx::MPI::rank(comm);
  const int mpi_size = dolfinx::MPI::size(comm);

  std::vector<std::uint8_t> has_cells(mpi_size, cell_indicator);

  // Get received data sizes from each rank
  std::vector<std::uint8_t> procs_with_cells(mpi_size, -1);
  MPI_Alltoall(has_cells.data(), 1, MPI_UINT8_T, procs_with_cells.data(), 1,
               MPI_UINT8_T, comm);
  std::vector<std::int32_t> edges;
  edges.reserve(mpi_size);
  // If current rank owns masters add all slaves as source edges
  if (procs_with_cells[mpi_rank] == 1)
    for (int i = 0; i < mpi_size; ++i)
      if ((i != mpi_rank) && (procs_with_cells[i] == 1))
        edges.push_back(i);
  const auto cell_map = mesh.topology().index_map(mesh.topology().dim());
  const std::int32_t num_local_cells = cell_map->size_local();
  std::vector<std::int32_t> num_dest_ranks(num_local_cells, 0);

  // Get map from remote processes to owned cells
  const dolfinx::graph::AdjacencyList<std::int32_t>& ghosted_indices
      = cell_map->scatter_fwd_indices();
  auto forward_comm
      = cell_map->comm(dolfinx::common::IndexMap::Direction::forward);
  std::array<std::vector<int>, 2> neighbours
      = dolfinx::MPI::neighbors(forward_comm);
  auto& src_ranks = neighbours[0];
  for (std::int32_t i = 0; i < ghosted_indices.num_nodes(); i++)
  {
    auto cells = ghosted_indices.links(i);
    for (auto cell : cells)
      num_dest_ranks[cell]++;
  }

  // Create map from owned cells to other processes having this cell as a ghost
  std::map<std::int32_t, std::set<int>> cell_to_procs;
  for (std::int32_t i = 0; i < ghosted_indices.num_nodes(); i++)
  {
    auto cells = ghosted_indices.links(i);
    for (auto cell : cells)
      cell_to_procs[cell].insert(src_ranks[i]);
  }
  for (auto cell : cells)
  {
    for (auto rank : edges)
      cell_to_procs[cell].insert(rank);
  }
  for (std::int32_t i = 0; i < num_local_cells; i++)
    cell_to_procs[i].insert(mpi_rank);

  // Create Adjacency list
  std::vector<std::size_t> num_shared_procs(num_local_cells, 0);
  for (auto proc_map : cell_to_procs)
    if (proc_map.first < num_local_cells)
      num_shared_procs[proc_map.first] = proc_map.second.size();
  std::vector<std::int32_t> offsets(num_local_cells + 1, 0);
  std::partial_sum(num_shared_procs.begin(), num_shared_procs.end(),
                   offsets.begin() + 1);
  std::vector<std::int32_t> data_ranks(offsets.back());
  std::fill(num_shared_procs.begin(), num_shared_procs.end(), 0);
  for (auto index : cell_to_procs)
  {
    for (auto rank : index.second)
    {
      if (index.first < num_local_cells)
        data_ranks[offsets[index.first] + num_shared_procs[index.first]++]
            = rank;
    }
  }
  dolfinx::graph::AdjacencyList<std::int32_t> new_shared_indices(
      std::move(data_ranks), std::move(offsets));
  return new_shared_indices;
  //   return update_ghosts(mesh, new_shared_indices);
};

} // namespace dolfinx_contact