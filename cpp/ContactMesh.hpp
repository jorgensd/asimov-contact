// Copyright (C) 2022 Jørgen S. Dokken and Igor Baratta
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>

namespace
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

  auto partitioner = [&dest](...) { return dest; };

  return dolfinx::mesh::create_mesh(mesh.comm(), cell_vertices, geometry.cmap(),
                                    x, dolfinx::mesh::GhostMode::shared_facet,
                                    partitioner);
}

} // namespace

namespace dolfinx_contact
{
dolfinx::mesh::Mesh add_ghost_cells(const dolfinx::mesh::Mesh& mesh,
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

  // Create an Adjacency list of all shared cells
  const auto cell_map = mesh.topology().index_map(mesh.topology().dim());
  const std::int32_t num_local_cells = cell_map->size_local();
  std::vector<std::int32_t> num_dest_ranks(num_local_cells, 0);
  std::map<std::int32_t, std::set<int>> shared_indices
      = cell_map->compute_shared_indices();
  for (auto cell : cells)
  {
    for (auto edge : edges)
    {
      if (cell < num_local_cells)
        shared_indices[cell].insert(edge);
    }
  }
  std::vector<std::size_t> num_shared_processes(num_local_cells, 0);
  for (auto cell : shared_indices)
  {
    if (cell.first < num_local_cells)
      num_shared_processes[cell.first] = cell.second.size();
  }
  std::vector<std::int32_t> offsets(num_local_cells + 1, 0);
  std::partial_sum(num_shared_processes.begin(), num_shared_processes.end(),
                   offsets.begin() + 1);
  std::vector<std::int32_t> data_ranks(offsets.back());
  std::fill(num_shared_processes.begin(), num_shared_processes.end(), 0);
  for (auto index : shared_indices)
  {
    for (auto rank : index.second)
    {
      if (index.first < num_local_cells)
        data_ranks[offsets[index.first] + num_shared_processes[index.first]++]
            = rank;
    }
  }

  dolfinx::graph::AdjacencyList<std::int32_t> new_shared_indices(
      std::move(data_ranks), std::move(offsets));
  return update_ghosts(mesh, new_shared_indices);
};

} // namespace dolfinx_contact