// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "RayTracing.h"
#include "error_handling.h"
#include "geometric_quantities.h"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xio.hpp>

#include <xtensor/xtensor.hpp>
namespace dolfinx_contact
{

enum class ContactMode
{
  ClosestPoint,
  RayTracing
};

enum class Kernel
{
  Rhs,
  Jac,
  MeshTieRhs,
  MeshTieJac
};
// NOTE: this function should change signature to T * ,..... , num_links,
// num_dofs_per_link
template <typename T>
using kernel_fn
    = std::function<void(std::vector<std::vector<T>>&, xtl::span<const T>,
                         const T*, const double*, const int, const std::size_t,
                         const std::vector<std::int32_t>&)>;

/// This function computes the pull back for a set of points x on a cell
/// described by coordinate_dofs as well as the corresponding Jacobian, their
/// inverses and their determinants
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on physical element
/// @param[in ,out] X: pull pack of x (points on reference element)
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] cmap: the coordinate element
//-----------------------------------------------------------------------------
void pull_back(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
               xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
               xt::xtensor<double, 2>& X,
               const xt::xtensor<double, 2>& coordinate_dofs,
               const dolfinx::fem::CoordinateElement& cmap);

/// @param[in] cells: the cells to be sorted
/// @param[in, out] perm: the permutation for the sorted cells
/// @param[out] pair(unique_cells, offsets): unique_cells is a vector of
/// sorted cells with all duplicates deleted, offsets contains the start and
/// end for each unique value in the sorted vector with all duplicates
// Example: cells = [5, 7, 6, 5]
//          unique_cells = [5, 6, 7]
//          offsets = [0, 2, 3, 4]
//          perm = [0, 3, 2, 1]
// Then given a cell and its index ("i") in unique_cells, one can recover the
// indices for its occurance in cells with perm[k], where
// offsets[i]<=k<offsets[i+1]. In the example if i = 0, then perm[k] = 0 or
// perm[k] = 3.
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
sort_cells(const std::span<const std::int32_t>& cells,
           const std::span<std::int32_t>& perm);

/// @param[in] u: dolfinx function on function space base on basix element
/// @param[in] mesh: mesh to be updated
/// Adds perturbation u to mesh
void update_geometry(const dolfinx::fem::Function<PetscScalar>& u,
                     std::shared_ptr<dolfinx::mesh::Mesh> mesh);

/// Compute the positive restriction of a double, i.e. f(x)= x if x>0 else 0
double R_plus(double x);

/// Compute the negative restriction of a double, i.e. f(x)= x if x<0 else 0
double R_minus(double x);

/// Compute the derivative of the positive restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_plus(double x);

/// Compute the derivative of the negative restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_minus(double x);

/// Get shape of in,out variable for filling basis functions in for
/// evaluate_basis_functions
std::array<std::size_t, 4>
evaluate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                     const std::size_t num_points,
                     const std::size_t num_derivatives);

/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3).
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] basis_values The values at the points. Values are not
/// computed for points with a negative cell index. This argument must be passed
/// with the correct size (num_points, number_of_dofs, value_size).
void evaluate_basis_functions(const dolfinx::fem::FunctionSpace& V,
                              const xt::xtensor<double, 2>& x,
                              const std::span<const std::int32_t>& cells,
                              xt::xtensor<double, 4>& basis_values,
                              std::size_t num_derivatives);

/// Compute physical normal
/// @param[in] n_ref facet normal on reference element
/// @param[in] K inverse Jacobian
/// @param[in,out] n_phys facet normal on physical element
void compute_normal(const xt::xtensor<double, 1>& n_ref,
                    const xt::xtensor<double, 2>& K,
                    xt::xarray<double>& n_phys);

/// Compute the following jacobians on a given facet:
/// J: physical cell -> reference cell (and its inverse)
/// J_tot: physical facet -> reference facet
/// @param[in] q - index of quadrature points
/// @param[in,out] J - Jacboian between reference cell and physical cell
/// @param[in,out] K - inverse of J
/// @param[in,out] J_tot - J_f*J
/// @param[in] J_f - the Jacobian between reference facet and reference cell
/// @param[in] dphi - derivatives of coordinate basis tabulated for quardrature
/// points
/// @param[in] coords - the coordinates of the facet
/// @return absolute value of determinant of J_tot
double compute_facet_jacobians(std::size_t q, xt::xtensor<double, 2>& J,
                               xt::xtensor<double, 2>& K,
                               xt::xtensor<double, 2>& J_tot,
                               const xt::xtensor<double, 2>& J_f,
                               const xt::xtensor<double, 3>& dphi,
                               const xt::xtensor<double, 2>& coords);

/// @brief Convenience function to update Jacobians
///
/// For affine geometries, the input determinant is returned.
/// For non-affine geometries, the Jacobian, it's inverse and the total Jacobian
/// (J*J_f) is computed.
/// @param[in] cmap The coordinate element
std::function<double(
    std::size_t, double, xt::xtensor<double, 2>&, xt::xtensor<double, 2>&,
    xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&,
    const xt::xtensor<double, 3>&, const xt::xtensor<double, 2>&)>
get_update_jacobian_dependencies(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convenience function to update facet normals
///
/// For affine geometries, a do nothing function is returned.
/// For non-affine geometries, a function updating the physical facet normal is
/// returned.
/// @param[in] cmap The coordinate element
std::function<void(xt::xtensor<double, 1>&, const xt::xtensor<double, 2>&,
                   const xt::xtensor<double, 2>&, std::size_t)>
get_update_normal(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convert local entity indices to integration entities
///
/// Compute the active entities in DOLFINx format for a given integral type over
/// a set of entities If the integral type is cell, return the input, if it is
/// exterior facets, return a list of pairs (cell, local_facet_index), and if it
/// is interior facets, return a list of tuples (cell_0, local_facet_index_0,
/// cell_1, local_facet_index_1) for each entity.
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::vector<std::int32_t>
compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        std::span<const std::int32_t> entities,
                        dolfinx::fem::IntegralType integral);

/// @brief Compute the geometry dof indices for a set of entities
///
/// For a set of entities, compute the geometry closure dofs of the entity.
///
/// @param[in] mesh The mesh
/// @param[in] dim The dimension of the entities
/// @param[in] entities List of mesh entities
/// @returns An adjacency list where the i-th link corresponds to the
/// closure dofs of the i-th input entity
dolfinx::graph::AdjacencyList<std::int32_t>
entities_to_geometry_dofs(const mesh::Mesh& mesh, int dim,
                          const std::span<const std::int32_t>& entity_list);

/// @brief find candidate facets within a given radius of puppet facets
///
/// Given a list of puppet facets and a list of candidate facets return
/// only those candidate facet within the given radius
///
/// @param[in] mesh The mesh
/// @param[in] puppet_facets Puppet facets
/// @param[in] candidate_facets Candidate facets
/// @param[in] radius The search radius
/// @return candidate facets within radius of puppet facets
std::vector<std::int32_t> find_candidate_surface_segment(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    const std::vector<std::int32_t>& puppet_facets,
    const std::vector<std::int32_t>& candidate_facets, const double radius);

/// @brief compute physical points on set of facets
///
/// Given a list of facets and the basis functions evaluated at set of points on
/// reference facets compute physical points
///
/// @param[in] mesh The mesh
/// @param[in] facets The list of facets as (cell, local_facet). The data is
/// flattened row-major
/// @param[in] offsets for accessing the basis_values for local_facet
/// @param[in] phi Basis functions evaluated at desired set of point osn
/// reference facet
/// @param[in, out] qp_phys vector to stor physical points per facet
void compute_physical_points(const dolfinx::mesh::Mesh& mesh,
                             std::span<const std::int32_t> facets,
                             const std::vector<int>& offsets,
                             const xt::xtensor<double, 2>& phi,
                             std::vector<xt::xtensor<double, 2>>& qp_phys);

/// Compute the closest entity at every quadrature point on a subset of facets
/// on one mesh, to a subset of facets on the other mesh.
/// @param[in] quadrature_mesh The mesh to compute quadrature points on
/// @param[in] quadrature_facets The facets to compute quadrature points on,
/// defined as (cell, local_facet_index). Flattened row-major.
/// @param[in] candidate_mesh The mesh with the facets we want to compute the
/// distance to
/// @param[in] candidate_facets The facets on candidate_mesh,defined as (cell,
/// local_facet_index). Flattened row-major.
/// @param[in] q_rule The quadrature rule for the input facets
/// @param[in] mode The contact mode, either closest point or ray-tracing
/// @returns A tuple (closest_facets, reference_points) where `closest_facets`
/// is an adjacency list for each input facet in quadrature facets, where the
/// links indicate which facet on the other mesh is closest for each quadrature
/// point.`reference_points` is the corresponding points on the reference
/// element for each quadrature point.  Shape (num_facets, num_q_points, tdim).
/// Flattened to (num_facets*num_q_points, tdim).
std::pair<dolfinx::graph::AdjacencyList<std::int32_t>, xt::xtensor<double, 2>>
compute_distance_map(const dolfinx::mesh::Mesh& quadrature_mesh,
                     std::span<const std::int32_t> quadrature_facets,
                     const dolfinx::mesh::Mesh& candidate_mesh,
                     std::span<const std::int32_t> candidate_facets,
                     const QuadratureRule& q_rule, ContactMode mode);

/// Compute the relation between two meshes (mesh_q) and
/// (mesh_c) by computing the intersection of rays from
/// mesh_q onto mesh_c at a specific set of quadrature
/// points on a subset of facets. There is also a subset of
/// facets on mesh_c we use for intersection checks.
/// @param[in] quadrature_mesh The mesh to compute rays
/// from
/// @param[in] quadrature_facets Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @param[in] quadrature_points The quadrature points on
/// the `quadrature_facets`. The ith entry corresponds to
/// the quadrature points in the ith tuple of
/// `quadrature_facet`.
/// @param[in] candidate_mesh The mesh to compute ray
/// intersections with
/// @param[in] candidate_facets Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @returns A tuple (facet_map, reference_points), where
/// `facet_map` is an AdjacencyList from the ith facet
/// tuple in `quadrature_facets` to the facet (index local
/// to process) in `candidate_facets`. `reference_points`
/// are the reference points for the point of intersection
/// for each of the quadrature points on each facet. Shape
/// (num_facets, num_q_points, tdim). Flattened to
/// (num_facets*num_q_points, tdim).
template <std::size_t tdim, std::size_t gdim>
std::pair<dolfinx::graph::AdjacencyList<std::int32_t>, xt::xtensor<double, 2>>
compute_raytracing_map(const dolfinx::mesh::Mesh& quadrature_mesh,
                       xtl::span<const std::int32_t> quadrature_facets,
                       std::vector<xt::xtensor<double, 2>> quadrature_points,
                       const dolfinx::mesh::Mesh& candidate_mesh,
                       xtl::span<const std::int32_t> candidate_facets)
{
  dolfinx::common::Timer timer("~Raytracing");

  assert(candidate_mesh.geometry().dim() == gdim);
  assert(quadrature_mesh.geometry().dim() == gdim);
  assert(candidate_mesh.topology().dim() == tdim);
  assert(quadrature_mesh.topology().dim() == tdim);

  // Convert (cell, local facet index) into facet index
  // (local to process) Convert cell,local_facet_index to
  // facet_index (local to proc)
  std::vector<std::int32_t> facets(candidate_facets.size() / 2);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = candidate_mesh.topology().connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell->facet connectivity on candidate "
                             "mesh.");
  }

  for (std::size_t i = 0; i < candidate_facets.size(); i += 2)
  {
    auto local_facets = c_to_f->links(candidate_facets[i]);
    assert(!local_facets.empty());
    assert((std::size_t)candidate_facets[i + 1] < local_facets.size());
    facets[i / 2] = local_facets[candidate_facets[i + 1]];
  }

  // Structures used for computing physical normal
  xt::xtensor<double, 2> J({gdim, (std::size_t)tdim});
  xt::xtensor<double, 2> K({(std::size_t)tdim, gdim});
  const dolfinx::mesh::Geometry& geom_q = quadrature_mesh.geometry();
  const dolfinx::fem::CoordinateElement& cmap_q
      = quadrature_mesh.geometry().cmap();
  const dolfinx::mesh::Topology& top_q = quadrature_mesh.topology();
  xtl::span<const double> q_x = geom_q.x();
  const graph::AdjacencyList<std::int32_t>& q_dofmap = geom_q.dofmap();
  const std::size_t num_nodes_q = cmap_q.dim();
  xt::xtensor<double, 2> coordinate_dofs_q({num_nodes_q, gdim});
  auto [reference_normals, rn_shape] = basix::cell::facet_outward_normals(
      dolfinx::mesh::cell_type_to_basix_type(top_q.cell_type()));

  const std::array<std::size_t, 4> basis_shape_q = cmap_q.tabulate_shape(1, 1);
  xt::xtensor<double, 4> basis_values_q(basis_shape_q);
  xt::xtensor<double, 2> dphi_q({tdim, basis_shape_q[2]});

  // Structures used for raytracing
  dolfinx::mesh::CellType cell_type = candidate_mesh.topology().cell_type();
  const dolfinx::mesh::Geometry& c_geometry = candidate_mesh.geometry();
  const dolfinx::fem::CoordinateElement& cmap_c = c_geometry.cmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& c_dofmap
      = c_geometry.dofmap();
  xtl::span<const double> c_x = c_geometry.x();
  newton_storage<tdim, gdim> allocated_memory;
  const std::array<std::size_t, 4> basis_shape_c = cmap_c.tabulate_shape(1, 1);
  xt::xtensor<double, 4> basis_values_c(basis_shape_c);
  xt::xtensor<double, 2> dphi_c({tdim, basis_shape_c[2]});
  const std::size_t num_nodes_c = cmap_c.dim();
  xt::xtensor<double, 2> coordinate_dofs_c({num_nodes_c, gdim});

  // Variable to hold jth point for Jacbian computation
  error::check_cell_type(cell_type);
  xt::xtensor_fixed<double, xt::xshape<gdim>> normal;
  xt::xtensor_fixed<double, xt::xshape<gdim>> q_point;
  const std::size_t num_q_points = quadrature_points[0].shape(0);
  std::vector<std::int32_t> colliding_facet(
      quadrature_facets.size() / 2 * num_q_points, -1);
  xt::xtensor<double, 2> reference_points(
      {quadrature_facets.size() / 2 * num_q_points, tdim});
  for (std::size_t i = 0; i < quadrature_facets.size(); i += 2)
  {
    // Pack coordinate dofs
    auto x_dofs = q_dofmap.links(quadrature_facets[i]);
    assert(x_dofs.size() == num_nodes_q);
    for (std::size_t j = 0; j < num_nodes_q; ++j)
    {
      const int pos = 3 * x_dofs[j];
      dolfinx::common::impl::copy_N<gdim>(
          std::next(q_x.cbegin(), pos),
          std::next(coordinate_dofs_q.begin(), j * gdim));
    }
    const std::int32_t facet_index = quadrature_facets[i + 1];
    const xt::xtensor<double, 2>& facet_points = quadrature_points[i / 2];
    for (std::size_t j = 0; j < num_q_points; ++j)
    {

      // Compute inverse Jacobian for covariant Piola
      // transform
      cmap_q.tabulate(1,
                      xt::reshape_view(xt::row(quadrature_points[i / 2], j),
                                       {static_cast<std::size_t>(1), tdim}),
                      basis_values_q);
      dphi_q = xt::view(basis_values_q, xt::xrange(1, (int)tdim + 1), 0,
                        xt::all(), 0);
      std::fill(J.begin(), J.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                        coordinate_dofs_q, J);
      std::fill(K.begin(), K.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      // transform
      std::fill(normal.begin(), normal.end(), 0);
      physical_facet_normal(std::span(normal.data(), gdim), K,
                            std::span(std::next(reference_normals.begin(),
                                                rn_shape[1] * facet_index),
                                      rn_shape[1]));
      dolfinx::common::impl::copy_N<gdim>(
          std::next(facet_points.begin(), j * gdim), q_point.begin());
      std::size_t cell_idx = -1;
      allocated_memory.tangents = compute_tangents<gdim>(normal);
      allocated_memory.point = q_point;

      int status = 0;
      for (std::size_t c = 0; c < candidate_facets.size(); c += 2)
      {
        // Get cell geometry for candidate cell, reusing
        // coordinate dofs to store new coordinate
        auto x_dofs = c_dofmap.links(candidate_facets[c]);
        for (std::size_t k = 0; k < x_dofs.size(); ++k)
        {
          dolfinx::common::impl::copy_N<gdim>(
              std::next(c_x.begin(), 3 * x_dofs[k]),
              std::next(coordinate_dofs_c.begin(), gdim * k));
        }
        // Assign Jacobian of reference mapping
        allocated_memory.dxi = get_parameterization_jacobian<tdim>(
            cell_type, candidate_facets[c + 1]);
        // Get parameterization map
        auto reference_map
            = get_parameterization<tdim>(cell_type, candidate_facets[c + 1]);
        status = raytracing_cell<tdim, gdim>(
            allocated_memory, basis_values_c, dphi_c, 25, 1e-8, cmap_c,
            cell_type, coordinate_dofs_c, reference_map);
        if (status > 0)
        {
          cell_idx = c / 2;
          break;
        }
      }
      if (status > 0)
      {
        colliding_facet[i / 2 * num_q_points + j] = facets[cell_idx];
        xt::row(reference_points, i / 2 * num_q_points + j)
            = xt::row(allocated_memory.X_k, 0);
      }
    }
  }
  std::vector<std::int32_t> offset(quadrature_facets.size() / 2 + 1);
  std::iota(offset.begin(), offset.end(), 0);
  std::for_each(offset.begin(), offset.end(),
                [num_q_points](auto& i) { i *= num_q_points; });
  timer.stop();
  return {dolfinx::graph::AdjacencyList<std::int32_t>(colliding_facet, offset),
          reference_points};
}

/// Compute the relation between a set of points and a mesh by computing the
/// closest point on mesh at a specific set of points. There is also a subset
/// of facets on mesh we use for intersection checks.
/// @param[in] mesh The mesh to compute the closest point at
/// @param[in] facet_tuples Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @param[in] points The points to compute the closest entity from.
/// Shape (num_quadrature_points, 3). Flattened row-major
/// @returns A tuple (closest_facets, reference_points), where
/// `closest_entities[i]` is the closest entity in `facet_tuples` for the ith
/// input point
template <std::size_t tdim, std::size_t gdim>
std::pair<std::vector<std::int32_t>, xt ::xtensor<double, 2>>
compute_projection_map(const dolfinx::mesh::Mesh& mesh,
                       std::span<const std::int32_t> facet_tuples,
                       std::span<const double> points)
{
  assert(tdim == mesh.topology().dim());
  assert(mesh.geometry().dim() == gdim);

  const std::size_t num_points = points.size() / 3;

  // Convert cell,local_facet_index to facet_index (local
  // to proc)
  std::vector<std::int32_t> facets(facet_tuples.size() / 2);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell->facet connectivity on candidate "
                             "mesh.");
  }

  for (std::size_t i = 0; i < facet_tuples.size(); i += 2)
  {
    auto local_facets = c_to_f->links(facet_tuples[i]);
    assert(!local_facets.empty());
    assert((std::size_t)facet_tuples[i + 1] < local_facets.size());
    facets[i / 2] = local_facets[facet_tuples[i + 1]];
  }

  // Compute closest entity for each point
  dolfinx::geometry::BoundingBoxTree bbox(mesh, tdim - 1, facets);
  dolfinx::geometry::BoundingBoxTree midpoint_tree
      = dolfinx::geometry::create_midpoint_tree(mesh, tdim - 1, facets);
  std::vector<std::int32_t> closest_facets
      = dolfinx::geometry::compute_closest_entity(bbox, midpoint_tree, mesh,
                                                  points);

  xt::xtensor<double, 2> candidate_x({num_points, static_cast<std::size_t>(3)});
  xtl::span<const double> mesh_geometry = mesh.geometry().x();
  const dolfinx::fem::CoordinateElement& cmap = mesh.geometry().cmap();
  {
    // Find displacement vector from each point
    // to closest entity. As a point on the surface
    // might have penetrated the cell in question, we use
    // the convex hull of the surface facet for distance
    // computations

    // Get information aboute cell type and number of
    // closure dofs on the facet NOTE: Assumption that we
    // do not have variable facet types (prism/pyramid
    // cell)
    const dolfinx::fem::ElementDofLayout layout = cmap.create_dof_layout();

    error::check_cell_type(mesh.topology().cell_type());
    const std::vector<std::int32_t>& closure_dofs
        = layout.entity_closure_dofs(tdim - 1, 0);
    const std::size_t num_facet_dofs = closure_dofs.size();

    // Get the geometry dofs of closest facets
    const dolfinx::graph::AdjacencyList<std::int32_t> facets_geometry
        = dolfinx_contact::entities_to_geometry_dofs(mesh, tdim - 1,
                                                     closest_facets);
    assert(facets_geometry.num_nodes() == num_points);

    // Compute physical points for each facet
    std::vector<double> coordinate_dofs(3 * num_facet_dofs);
    for (std::size_t i = 0; i < num_points; ++i)
    {
      // Get the geometry dofs for the ith facet, qth
      // quadrature point
      auto candidate_facet_dofs = facets_geometry.links(i);
      assert(num_facet_dofs == candidate_facet_dofs.size());

      // Get the (geometrical) coordinates of the facets
      for (std::size_t l = 0; l < num_facet_dofs; ++l)
      {
        dolfinx::common::impl::copy_N<3>(
            std::next(mesh_geometry.begin(), 3 * candidate_facet_dofs[l]),
            std::next(coordinate_dofs.begin(), 3 * l));
      }

      // Compute distance between convex hull of facet and point
      std::array<double, 3> dist_vec = dolfinx::geometry::compute_distance_gjk(
          coordinate_dofs, std::span(std::next(points.begin(), 3 * i), 3));

      // Compute point on closest facet
      for (std::size_t l = 0; l < 3; ++l)
        candidate_x(i, l) = points[3 * i + l] + dist_vec[l];
    }
  }

  // Pull back to reference point for each facet on the surface
  xt::xtensor<double, 2> candidate_X({num_points, (std::size_t)gdim});
  {
    // Temporary data structures used in loop over each
    // quadrature point on each facet
    xt::xtensor<double, 3> J(
        {static_cast<std::size_t>(1), (std::size_t)gdim, (std::size_t)tdim});
    xt::xtensor<double, 3> K(
        {static_cast<std::size_t>(1), (std::size_t)gdim, (std::size_t)tdim});
    xt::xtensor<double, 1> detJ({static_cast<std::size_t>(1)});
    xt::xtensor<double, 2> x({static_cast<std::size_t>(1), (std::size_t)tdim});
    xt::xtensor<double, 2> X({static_cast<std::size_t>(1), (std::size_t)gdim});
    const std::size_t num_dofs_g = cmap.dim();
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh.geometry().dofmap();
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
    auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
    if (!f_to_c)
      throw std::runtime_error("Missing facet to cell connectivity");
    for (std::size_t i = 0; i < closest_facets.size(); ++i)
    {
      // Get cell connected to facet
      auto cells = f_to_c->links(closest_facets[i]);
      assert(cells.size() == 1);

      // Pack coordinate dofs
      auto x_dofs = x_dofmap.links(cells.front());
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        dolfinx::common::impl::copy_N<gdim>(
            std::next(mesh_geometry.begin(), 3 * x_dofs[j]),
            std::next(coordinate_dofs.begin(), j * gdim));
      }

      // Copy closest point in physical space
      std::fill(x.begin(), x.end(), 0);
      dolfinx::common::impl::copy_N<gdim>(std::next(candidate_x.begin(), 3 * i),
                                          x.begin());

      // NOTE: Would benefit from pulling back all points
      // in a single cell at the same time
      // Pull back coordinates
      std::fill(J.begin(), J.end(), 0);
      std::fill(K.begin(), K.end(), 0);
      dolfinx_contact::pull_back(J, K, detJ, x, X, coordinate_dofs, cmap);

      // Copy into output
      dolfinx::common::impl::copy_N<tdim>(
          X.begin(), std::next(candidate_X.begin(), i * tdim));
    }
  }
  return {closest_facets, candidate_x};
}
} // namespace dolfinx_contact