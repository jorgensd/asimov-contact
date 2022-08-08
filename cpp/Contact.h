// Copyright (C) 2021-2022 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "KernelData.h"
#include "QuadratureRule.h"
#include "SubMesh.h"
#include "elasticity.h"
#include "geometric_quantities.h"
#include "meshtie_kernels.h"
#include "utils.h"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>

using mat_set_fn = const std::function<int(
    const std::span<const std::int32_t>&, const std::span<const std::int32_t>&,
    const std::span<const PetscScalar>&)>;

namespace dolfinx_contact
{

namespace impl
{
/// Tabulate the coordinate element basis functions at quadrature points
///
/// @param[in] cmap The coordinate element
/// @param[in] q_rule The quadrature rule
std::pair<std::vector<double>, std::array<std::size_t, 4>>
tabulate(const dolfinx::fem::CoordinateElement& cmap,
         std::shared_ptr<const dolfinx_contact::QuadratureRule> q_rule)
{

  // Create quadrature points on reference facet
  const std::vector<double>& q_weights = q_rule->weights();
  const std::vector<double>& q_points = q_rule->points();
  assert(q_weights.size() == (std::size_t)q_rule->offset().back());
  // Tabulate Coordinate element (first derivative to compute Jacobian)
  std::array<std::size_t, 4> cmap_shape
      = cmap.tabulate_shape(0, q_weights.size());
  std::vector<double> cmap_basis(
      std::reduce(cmap_shape.begin(), cmap_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(0, q_points, {q_weights.size(), q_rule->tdim()}, cmap_basis);
  return {cmap_basis, cmap_shape};
}
} // namespace impl

class Contact
{
public:
  /// Constructor
  /// @param[in] markers List of meshtags defining the contact surfaces
  /// @param[in] surfaces Adjacency list. Links of i contains meshtag values
  /// associated with ith meshtag in markers
  /// @param[in] contact_pairs list of pairs (i, j) marking the ith and jth
  /// surface in surfaces->array() as a contact pair
  /// @param[in] V The functions space
  /// @param[in] q_deg The quadrature degree.
  Contact(const std::vector<
              std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
          std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
              surfaces,
          const std::vector<std::array<int, 2>>& contact_pairs,
          std::shared_ptr<dolfinx::fem::FunctionSpace> V, const int q_deg = 3,
          ContactMode mode = ContactMode::ClosestPoint);

  /// Return meshtag value for surface with index surface
  /// @param[in] surface - the index of the surface
  int surface_mt(int surface) const { return _surfaces[surface]; }

  /// Return contact pair
  /// @param[in] pair - the index of the contact pair
  const std::array<int, 2>& contact_pair(int pair) const
  {
    return _contact_pairs[pair];
  }

  // Return active entities for surface s
  const std::vector<std::int32_t>& active_entities(int s) const
  {
    return _cell_facet_pairs[s];
  }
  // set quadrature rule
  void set_quadrature_rule(QuadratureRule q_rule)
  {
    _quadrature_rule = std::make_shared<QuadratureRule>(q_rule);
  }

  /// return size of coefficients vector per facet on s
  /// @param[in] meshtie - Type of constraint,meshtie if true, unbiased contact
  /// if false
  std::size_t coefficients_size(bool meshtie);

  /// return distance map (adjacency map mapping quadrature points on surface
  /// to closest facet on other surface)
  /// @param[in] surface - index of the surface
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  facet_map(int surface) const
  {
    return _facet_maps[surface];
  }

  /// Return the quadrature points on physical facet for each facet on surface
  /// @param[in] surface The index of the surface (0 or 1).
  std::pair<std::vector<double>, std::array<std::size_t, 3>>
  qp_phys(int surface)
  {
    const std::size_t num_facets = _cell_facet_pairs[surface].size() / 2;
    const std::size_t num_q_points
        = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
    const std::size_t gdim = _V->mesh()->geometry().dim();
    std::array<std::size_t, 3> shape = {num_facets, num_q_points, gdim};
    return {_qp_phys[surface], shape};
  }

  /// Return the submesh corresponding to surface
  /// @param[in] surface The index of the surface (0 or 1).
  const SubMesh& submesh(int surface) const { return _submeshes[surface]; }
  // Return mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh() const { return _V->mesh(); }
  /// @brief Create a PETSc matrix with contact sparsity pattern
  ///
  /// Create a PETSc matrix with the sparsity pattern of the input form and the
  /// coupling contact interfaces
  ///
  /// @param[in] The bilinear form
  /// @param[in] The matrix type, see:
  /// https://petsc.org/main/docs/manualpages/Mat/MatType.html#MatType for
  /// available types
  /// @returns Mat The PETSc matrix
  Mat create_petsc_matrix(const dolfinx::fem::Form<PetscScalar>& a,
                          const std::string& type);

  /// Assemble matrix over exterior facets (for contact facets)
  /// @param[in] mat_set the function for setting the values in the matrix
  /// @param[in] bcs List of Dirichlet BCs
  /// @param[in] pair index of contact pair
  /// @param[in] kernel The integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_matrix(
      const mat_set_fn& mat_set,
      const std::vector<
          std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
      int pair, const kernel_fn<PetscScalar>& kernel,
      const std::span<const PetscScalar> coeffs, int cstride,
      const std::span<const PetscScalar>& constants);

  /// Assemble vector over exterior facet (for contact facets)
  /// @param[in] b The vector
  /// @param[in] pair index of contact pair
  /// @param[in] kernel The integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_vector(std::span<PetscScalar> b, int pair,
                       const kernel_fn<PetscScalar>& kernel,
                       const std::span<const PetscScalar>& coeffs, int cstride,
                       const std::span<const PetscScalar>& constants);

  /// @brief Generate contact kernel
  ///
  /// The kernel will expect input on the form
  /// @param[in] type The kernel type (Either `Jac` or `Rhs`).
  /// @returns Kernel function that takes in a vector (b) to assemble into, the
  /// coefficients (`c`), the constants (`w`), the local facet entity (`entity
  /// _local_index`), the quadrature permutation and the number of cells on the
  /// other contact boundary coefficients are extracted from.
  /// @note The ordering of coefficients are expected to be `mu`, `lmbda`, `h`,
  /// `gap`, `normals` `test_fn`, `u`, `u_opposite`.
  /// @note The scalar valued coefficients `mu`,`lmbda` and `h` are expected to
  /// be DG-0 functions, with a single value per facet.
  /// @note The coefficients `gap`, `normals`,`test_fn` and `u_opposite` is
  /// packed at quadrature points. The coefficient `u` is packed at dofs.
  /// @note The vector valued coefficents `gap`, `test_fn`, `u`, `u_opposite`
  /// has dimension `bs == gdim`.
  kernel_fn<PetscScalar> generate_kernel(Kernel type)
  {

    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
    assert(mesh);
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const std::size_t bs = _V->dofmap()->bs();
    // FIXME: This will not work for prism meshes
    const std::vector<std::size_t>& qp_offsets = _quadrature_rule->offset();
    const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
    const std::size_t max_links
        = *std::max_element(_max_links.begin(), _max_links.end());
    const std::size_t ndofs_cell
        = _V->dofmap()->element_dof_layout().num_dofs();

    // Coefficient offsets
    // Expecting coefficients in following order:
    // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
    std::vector<std::size_t> cstrides
        = {1,
           1,
           1,
           num_q_points * gdim,
           num_q_points * gdim,
           num_q_points * ndofs_cell * bs * max_links,
           ndofs_cell * bs,
           num_q_points * bs};

    auto kd = dolfinx_contact::KernelData(_V, _quadrature_rule, cstrides);

    /// @brief Assemble kernel for RHS of unbiased contact problem
    ///
    /// Assemble of the residual of the unbiased contact problem into vector
    /// `b`.
    /// @param[in,out] b The vector to assemble the residual into
    /// @param[in] c The coefficients used in kernel. Assumed to be
    /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
    /// @param[in] w The constants used in kernel. Assumed to be ordered as
    /// `gamma`, `theta`.
    /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
    /// be padded to 3D, (shape (num_nodes, 3)).
    /// @param[in] facet_index Local facet index (relative to cell)
    /// @param[in] num_links How many cells from opposite surface are connected
    /// with the cell.
    /// @param[in] q_indices The quadrature points to loop over
    kernel_fn<PetscScalar> unbiased_rhs
        = [kd, gdim, ndofs_cell,
           bs](std::vector<std::vector<PetscScalar>>& b,
               std::span<const PetscScalar> c, const PetscScalar* w,
               const double* coordinate_dofs, const int facet_index,
               const std::size_t num_links,
               std::span<const std::int32_t> q_indices)

    {
      // Retrieve some data from kd
      std::array<std::size_t, 2> q_offset
          = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
      const std::uint32_t tdim = kd.tdim();

      // NOTE: DOLFINx has 3D input coordinate dofs
      cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

      // Create data structures for jacobians
      // We allocate more memory than required, but its better for the compiler
      std::array<double, 9> Jb;
      mdspan2_t J(Jb.data(), gdim, tdim);
      std::array<double, 9> Kb;
      mdspan2_t K(Kb.data(), tdim, gdim);
      std::array<double, 6> J_totb;
      mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
      double detJ = 0;
      std::array<double, 18> detJ_scratch;

      // Normal vector on physical facet at a single quadrature point
      std::array<double, 3> n_phys;

      // Pre-compute jacobians and normals for affine meshes
      if (kd.affine())
      {
        detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                               detJ_scratch, coord);
        physical_facet_normal(std::span(n_phys.data(), gdim), K,
                              stdex::submdspan(kd.facet_normals(), facet_index,
                                               stdex::full_extent));
      }

      // Extract constants used inside quadrature loop
      double gamma = c[2] / w[0];     // h/gamma
      double gamma_inv = w[0] / c[2]; // gamma/h
      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];
      // Extract reference to the tabulated basis function
      s_cmdspan2_t phi = kd.phi();
      s_cmdspan3_t dphi = kd.dphi();

      // Extract reference to quadrature weights for the local facet
      std::span<const double> _weights(kd.q_weights());
      const std::size_t num_points = q_offset.back() - q_offset.front();
      auto weights = _weights.subspan(q_offset.front(), num_points);

      // Temporary data structures used inside quadrature loop
      std::array<double, 3> n_surf = {0, 0, 0};
      std::vector<double> epsnb(ndofs_cell * gdim, 0);
      mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
      std::vector<double> trb(ndofs_cell * gdim, 0);
      mdspan2_t tr(trb.data(), ndofs_cell, gdim);

      // Loop over quadrature points
      for (auto q : q_indices)
      {
        const std::size_t q_pos = q_offset[0] + q;

        // Update Jacobian and physical normal
        detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                  detJ_scratch, coord);
        kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);
        double n_dot = 0;
        double gap = 0;
        // For ray tracing the gap is given by n * (Pi(x) -x)
        // where n = n_x
        // For closest point n = -n_y
        for (std::size_t i = 0; i < gdim; i++)
        {
          n_surf[i] = -c[kd.offsets(4) + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
          gap += c[kd.offsets(3) + q * gdim + i] * n_surf[i];
        }

        compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                    std::span(n_phys.data(), gdim), q_pos);
        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double jump_un = 0;
        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = kd.offsets(6) + i * bs;
          for (std::size_t j = 0; j < bs; j++)
          {
            PetscScalar coeff = c[block_index + j];
            tr_u += coeff * tr(i, j);
            epsn_u += coeff * epsn(i, j);
            jump_un += coeff * phi(q_pos, i) * n_surf[j];
          }
        }
        std::size_t offset_u_opp = kd.offsets(7) + q * bs;
        for (std::size_t j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf[j];
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        const double w0 = weights[q] * detJ;

        double Pn_u = R_plus((jump_un - gap) - gamma * sign_u) * w0;
        // Fill contributions of facet with itself
        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          for (std::size_t n = 0; n < bs; n++)
          {
            double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
            double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
            // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
            double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
            b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;

            // entries corresponding to v on the other surface
            for (std::size_t k = 0; k < num_links; k++)
            {
              std::size_t index = kd.offsets(5)
                                  + k * num_points * ndofs_cell * bs
                                  + i * num_points * bs + q * bs + n;
              double v_n_opp = c[index] * n_surf[n];

              b[k + 1][n + i * bs] -= 0.5 * gamma_inv * v_n_opp * Pn_u;
            }
          }
        }
      }
    };

    /// @brief Assemble kernel for Jacobian (LHS) of unbiased contact
    /// problem
    ///
    /// Assemble of the residual of the unbiased contact problem into matrix
    /// `A`.
    /// @param[in,out] A The matrix to assemble the Jacobian into
    /// @param[in] c The coefficients used in kernel. Assumed to be
    /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
    /// @param[in] w The constants used in kernel. Assumed to be ordered as
    /// `gamma`, `theta`.
    /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
    /// to be padded to 3D, (shape (num_nodes, 3)).
    /// @param[in] facet_index Local facet index (relative to cell)
    /// @param[in] num_links How many cells from opposite surface are connected
    /// with the cell.
    /// @param[in] q_indices The quadrature points to loop over
    kernel_fn<PetscScalar> unbiased_jac
        = [kd, gdim, ndofs_cell, bs](std::vector<std::vector<PetscScalar>>& A,
                                     std::span<const double> c, const double* w,
                                     const double* coordinate_dofs,
                                     const int facet_index,
                                     const std::size_t num_links,
                                     std::span<const std::int32_t> q_indices)
    {
      // Retrieve some data from kd
      std::array<std::size_t, 2> q_offset
          = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
      const std::uint32_t tdim = kd.tdim();

      // NOTE: DOLFINx has 3D input coordinate dofs
      cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

      // Create data structures for jacobians
      // We allocate more memory than required, but its better for the compiler
      std::array<double, 9> Jb;
      mdspan2_t J(Jb.data(), gdim, tdim);
      std::array<double, 9> Kb;
      mdspan2_t K(Kb.data(), tdim, gdim);
      std::array<double, 6> J_totb;
      mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
      double detJ;
      std::array<double, 18> detJ_scratch;

      // Normal vector on physical facet at a single quadrature point
      std::array<double, 3> n_phys;

      // Pre-compute jacobians and normals for affine meshes
      if (kd.affine())
      {
        detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                               detJ_scratch, coord);
        physical_facet_normal(std::span(n_phys.data(), gdim), K,
                              stdex::submdspan(kd.facet_normals(), facet_index,
                                               stdex::full_extent));
      }

      // Extract scaled gamma (h/gamma) and its inverse
      double gamma = c[2] / w[0];
      double gamma_inv = w[0] / c[2];

      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];

      cmdspan3_t dphi = kd.dphi();
      cmdspan2_t phi = kd.phi();
      std::span<const double> _weights(kd.q_weights());
      const std::size_t num_points = q_offset.back() - q_offset.front();
      auto weights = _weights.subspan(q_offset.front(), num_points);
      std::array<double, 3> n_surf = {0, 0, 0};
      std::vector<double> epsnb(ndofs_cell * gdim);
      mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
      std::vector<double> trb(ndofs_cell * gdim);
      mdspan2_t tr(trb.data(), ndofs_cell, gdim);

      // Loop over quadrature points
      for (auto q : q_indices)
      {
        const std::size_t q_pos = q_offset.front() + q;
        // Update Jacobian and physical normal
        detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                  detJ_scratch, coord);
        kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

        double n_dot = 0;
        double gap = 0;
        // The gap is given by n * (Pi(x) -x)
        // For raytracing n = n_x
        // For closest point n = -n_y
        for (std::size_t i = 0; i < gdim; i++)
        {
          n_surf[i] = -c[kd.offsets(4) + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
          gap += c[kd.offsets(3) + q * gdim + i] * n_surf[i];
        }

        compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                    std::span(n_phys.data(), gdim), q_pos);

        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double jump_un = 0;

        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = kd.offsets(6) + i * bs;
          for (std::size_t j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + j] * epsn(i, j);
            jump_un += c[block_index + j] * phi(q_pos, i) * n_surf[j];
          }
        }
        std::size_t offset_u_opp = kd.offsets(7) + q * bs;
        for (std::size_t j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf[j];
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        double Pn_u = dR_plus((jump_un - gap) - gamma * sign_u);

        // Fill contributions of facet with itself
        const double w0 = weights[q] * detJ;
        for (std::size_t j = 0; j < ndofs_cell; j++)
        {
          for (std::size_t l = 0; l < bs; l++)
          {
            double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
            double Pn_du
                = (phi(q_pos, j) * n_surf[l] - gamma * sign_du) * Pn_u * w0;

            sign_du *= w0;
            for (std::size_t i = 0; i < ndofs_cell; i++)
            {
              for (std::size_t b = 0; b < bs; b++)
              {
                double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
                double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
                double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * Pn_du * Pn_v;

                // entries corresponding to u and v on the other surface
                for (std::size_t k = 0; k < num_links; k++)
                {
                  std::size_t index = kd.offsets(5)
                                      + k * num_points * ndofs_cell * bs
                                      + j * num_points * bs + q * bs + l;
                  double du_n_opp = c[index] * n_surf[l];

                  du_n_opp *= w0 * Pn_u;
                  index = kd.offsets(5) + k * num_points * ndofs_cell * bs
                          + i * num_points * bs + q * bs + b;
                  double v_n_opp = c[index] * n_surf[b];
                  A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * du_n_opp * Pn_v;
                  A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * Pn_du * v_n_opp;
                  A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      += 0.5 * gamma_inv * du_n_opp * v_n_opp;
                }
              }
            }
          }
        }
      }
    };
    switch (type)
    {
    case Kernel::Rhs:
      return unbiased_rhs;
    case Kernel::Jac:
      return unbiased_jac;
    case Kernel::MeshTieRhs:
    {

      return generate_meshtie_kernel(type, _V, _quadrature_rule, max_links);
    }
    case Kernel::MeshTieJac:
    {
      return generate_meshtie_kernel(type, _V, _quadrature_rule, max_links);
    }
    default:
      throw std::invalid_argument("Unrecognized kernel");
    }
  }

  /// Compute push forward of quadrature points _qp_ref_facet to the
  /// physical facet for each facet in _facet_"origin_meshtag" Creates and
  /// fills _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose the surface
  void create_q_phys(int origin_meshtag)
  {
    // Get information depending on surface
    const SubMesh& submesh = _submeshes[origin_meshtag];

    const std::vector<std::int32_t> submesh_facets
        = submesh.get_submesh_tuples(_cell_facet_pairs[origin_meshtag]);
    auto mesh_sub = submesh.mesh();
    const std::size_t gdim = mesh_sub->geometry().dim();
    const std::vector<size_t>& qp_offsets = _quadrature_rule->offset();
    _qp_phys[origin_meshtag].resize((qp_offsets[1] - qp_offsets[0])
                                    * (submesh_facets.size() / 2) * gdim);
    compute_physical_points(
        *mesh_sub, submesh_facets, qp_offsets,
        cmdspan4_t(_reference_basis.data(), _reference_shape),
        _qp_phys[origin_meshtag]);
  }

  /// Compute maximum number of links
  /// I think this should actually be part of create_distance_map
  /// which should be easier after the rewrite of contact
  /// It is therefore called inside create_distance_map
  void max_links(int pair)
  {
    std::size_t max_links = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    const std::array<int, 2>& contact_pair = _contact_pairs[pair];
    const std::vector<std::int32_t>& active_facets
        = _cell_facet_pairs[contact_pair.front()];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
        = _facet_maps[pair];
    assert(map);
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = _submeshes[contact_pair.back()].facet_map();
    assert(facet_map);
    for (std::size_t i = 0; i < active_facets.size(); i += 2)
    {
      std::vector<std::int32_t> linked_cells;
      for (auto link : map->links((int)i / 2))
      {
        if (link >= 0)
        {
          auto facet_pair = facet_map->links(link);
          linked_cells.push_back(facet_pair.front());
        }
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      max_links = std::max(max_links, linked_cells.size());
    }
    _max_links[pair] = max_links;
  }

  /// For a given contact pair, for quadrature point on the first surface
  /// compute the closest candidate facet on the second surface.
  /// @param[in] pair The index of the contact pair
  /// @note This function alters _facet_maps[pair], _max_links[pair],
  /// _qp_phys, _phi_ref_facets
  void create_distance_map(int pair);

  /// Compute and pack the gap function for each quadrature point the set of
  /// facets. For a set of facets; go through the quadrature points on each
  /// facet find the closest facet on the other surface and compute the
  /// distance vector
  /// @param[in] pair - surface on which to integrate
  /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j]
  /// contains the jth component of the Gap on the ith facet at kth
  /// quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap(int pair)
  {
    // FIXME: This function should take in the quadrature points
    // (push_forward_quadrature) of the relevant facet, and the reference
    // points on the other surface (output of distance map)
    auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

    const std::shared_ptr<const dolfinx::mesh::Mesh>& quadrature_mesh
        = _submeshes[quadrature_mt].mesh();
    assert(quadrature_mesh);
    // Get (cell, local_facet_index) tuples on quadrature submesh
    const std::vector<std::int32_t> quadrature_facets
        = _submeshes[quadrature_mt].get_submesh_tuples(
            _cell_facet_pairs[quadrature_mt]);

    const std::shared_ptr<const dolfinx::mesh::Mesh>& candidate_mesh
        = _submeshes[candidate_mt].mesh();
    assert(candidate_mesh);
    // Get (cell, local_facet_index) tuples on candidate submesh
    const std::vector<std::int32_t> candidate_facets
        = _submeshes[candidate_mt].get_submesh_tuples(
            _cell_facet_pairs[candidate_mt]);

    auto [candidate_map, reference_x, shape] = compute_distance_map(
        *quadrature_mesh, quadrature_facets, *candidate_mesh, candidate_facets,
        *_quadrature_rule, _mode);

    // NOTE: Assumes same number of quadrature points on all facets
    error::check_cell_type(candidate_mesh->topology().cell_type());
    const std::size_t num_q_point
        = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
    const std::size_t num_facets = _cell_facet_pairs[quadrature_mt].size() / 2;
    const int q_gdim = quadrature_mesh->geometry().dim();
    mdspan3_t qp_span(_qp_phys[quadrature_mt].data(), num_facets, num_q_point,
                      q_gdim);

    // Get information about submesh geometry and topology
    const dolfinx::mesh::Geometry& geometry = candidate_mesh->geometry();
    const int gdim = geometry.dim();
    std::span<const double> x_g = geometry.x();
    auto x_dofmap = geometry.dofmap();
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const std::size_t num_dofs_g = cmap.dim();
    const dolfinx::mesh::Topology& topology = candidate_mesh->topology();
    const int tdim = topology.dim();

    std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
    cmdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);
    std::array<double, 3> coordb;
    mdspan2_t coord(coordb.data(), 1, gdim);

    // Pack gap function for each quadrature point on each facet
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = (int)num_q_point * gdim;

    auto f_to_c = candidate_mesh->topology().connectivity(tdim - 1, tdim);
    if (!f_to_c)
    {
      throw std::runtime_error("Missing facet to cell connectivity on "
                               "candidate submesh");
    }
    const std::array<std::size_t, 4> basis_shape
        = cmap.tabulate_shape(0, shape[0]);
    assert(basis_shape.back() == 1);
    std::vector<double> cmap_basis(std::reduce(
        basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(0, reference_x, shape, cmap_basis);
    cmdspan4_t full_basis(cmap_basis.data(), basis_shape);
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      int offset = (int)i * cstride;
      auto facets = candidate_map.links(i);
      assert(facets.size() == num_q_point);

      for (std::size_t q = 0; q < num_q_point; ++q)
      {

        // Skip negative facet indices (No facet on opposite surface has
        // been found)
        if (facets[q] < 0)
          continue;

        auto candidate_cells = f_to_c->links(facets[q]);
        assert(candidate_cells.size() == 1);

        // Copy coordinate dofs of candidate cell
        // Get cell geometry (coordinate dofs)
        auto x_dofs = x_dofmap.links(candidate_cells.front());
        assert(x_dofs.size() == num_dofs_g);
        for (std::size_t j = 0; j < num_dofs_g; ++j)
        {
          std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                      std::next(coordinate_dofsb.begin(), j * gdim));
        }

        auto basis_q = stdex::submdspan(
            full_basis, 0,
            std::pair{i * num_q_point + q, i * num_q_point + q + 1},
            stdex::full_extent, 0);

        dolfinx::fem::CoordinateElement::push_forward(coord, coordinate_dofs,
                                                      basis_q);
        for (int k = 0; k < gdim; k++)
          c[offset + q * gdim + k] = coordb[k] - qp_span(i, q, k);
      }
    }
    return {std::move(c), cstride};
  }

  /// Compute test functions on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_test_functions(int pair)
  {
    auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

    // Get mesh info for candidate side
    const std::shared_ptr<const dolfinx::mesh::Mesh>& candidate_mesh
        = _submeshes[candidate_mt].mesh();
    assert(candidate_mesh);
    const std::shared_ptr<const dolfinx::mesh::Mesh>& quadrature_mesh
        = _submeshes[quadrature_mt].mesh();
    assert(quadrature_mesh);

    // Get (cell, local_facet_index) tuples on quadrature submesh
    const std::vector<std::int32_t> quadrature_facets
        = _submeshes[quadrature_mt].get_submesh_tuples(
            _cell_facet_pairs[quadrature_mt]);

    // Get (cell, local_facet_index) tuples on candidate submesh
    const std::vector<std::int32_t> candidate_facets
        = _submeshes[candidate_mt].get_submesh_tuples(
            _cell_facet_pairs[candidate_mt]);

    auto [candidate_map, reference_x, shape]
        = dolfinx_contact::compute_distance_map(
            *quadrature_mesh, quadrature_facets, *candidate_mesh,
            candidate_facets, *_quadrature_rule, _mode);

    // Compute values of basis functions for all y = Pi(x) in qp
    auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace>(
        _submeshes[candidate_mt].create_functionspace(_V));

    std::shared_ptr<const dolfinx::fem::FiniteElement> element
        = V_sub->element();
    std::array<std::size_t, 4> b_shape
        = element->basix_element().tabulate_shape(0, shape[0]);
    if (b_shape.back() > 1)
      throw std::invalid_argument("pack_test_functions assumes values size 1");
    std::vector<double> basis_valuesb(
        std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
    element->tabulate(basis_valuesb, reference_x, shape, 0);
    cmdspan4_t basis_values(basis_valuesb.data(), b_shape);

    // Need to apply push forward and dof transformations to test functions
    assert((b_shape.front() == 1) and (b_shape.back() == 1));

    const basix::FiniteElement& b_el = element->basix_element();
    if (element->needs_dof_transformations()
        or b_el.map_type() != basix::maps::type::identity)
    {
      // If we want to do this we need to apply transformation and push
      // forward
      throw std::runtime_error(
          "Packing basis (test) functions of space that uses "
          "non-indentity maps is not supported");
    }

    // Convert facet index on candidate mesh into cell index
    const dolfinx::mesh::Topology& topology = candidate_mesh->topology();
    const int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    const std::vector<std::int32_t>& facets = candidate_map.array();
    error::check_cell_type(topology.cell_type());
    const std::size_t num_q_points
        = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
    const std::size_t num_facets = quadrature_facets.size() / 2;
    assert(num_facets * num_q_points == facets.size());
    std::vector<std::int32_t> cells(facets.size(), -1);
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
      if (facets[i] < 0)
        continue;
      auto f_cells = f_to_c->links(facets[i]);
      assert(f_cells.size() == 1);
      cells[i] = f_cells.front();
    }
    // FIXME: Aim to remove this as it depends on the state of the contact
    // algorithm
    const std::size_t max_links
        = *std::max_element(_max_links.begin(), _max_links.end());

    const std::size_t bs = element->block_size();
    const auto cstride = int(num_q_points * max_links * b_shape[2] * bs);
    std::vector<PetscScalar> cb(
        num_facets * max_links * num_q_points * b_shape[2] * bs, 0.0);
    stdex::mdspan<PetscScalar, stdex::dextents<std::size_t, 5>> c(
        cb.data(), num_facets, max_links, b_shape[2], num_q_points, bs);

    auto cell_imap = topology.index_map(tdim);
    const int num_local_cells
        = cell_imap->size_local() + cell_imap->num_ghosts();
    std::vector<std::int32_t> perm(num_q_points);
    for (std::size_t i = 0; i < c.extent(0); ++i)
    {
      std::span<const std::int32_t> f_cells(cells.data() + i * num_q_points,
                                            num_q_points);
      auto [unique_cells, offsets] = sort_cells(f_cells, perm);
      for (std::size_t j = 0; j < unique_cells.size(); ++j)
      {
        std::int32_t linked_cell = unique_cells[j];
        assert(linked_cell < num_local_cells);
        auto indices
            = std::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);

        assert(perm.size() >= (std::size_t)offsets[j + 1]);
        for (std::size_t k = 0; k < c.extent(2); ++k)
          for (std::size_t q = 0; q < indices.size(); ++q)
            for (std::size_t l = 0; l < c.extent(4); ++l)
            {
              c(i, j, k, indices[q], l)
                  = basis_values(0, i * num_q_points + indices[q], k, 0);
            }
      }
    }

    return {std::move(cb), cstride};
  }
  /// Compute gradient of test functions on opposite surface (initial
  /// configuration) at quadrature points of facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[in] u_packed -u packed on opposite surface per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_grad_test_functions(int pair, const std::span<const PetscScalar>& gap,
                           const std::span<const PetscScalar>& u_packed);

  /// Compute function on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_u_contact(int pair,
                 std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u)
  {
    dolfinx::common::Timer t("Pack contact u");
    auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

    // Get mesh info for candidate side
    const std::shared_ptr<const dolfinx::mesh::Mesh>& candidate_mesh
        = _submeshes[candidate_mt].mesh();
    assert(candidate_mesh);
    const std::shared_ptr<const dolfinx::mesh::Mesh>& quadrature_mesh
        = _submeshes[quadrature_mt].mesh();
    assert(quadrature_mesh);

    // Get (cell, local_facet_index) tuples on quadrature submesh
    const std::vector<std::int32_t> quadrature_facets
        = _submeshes[quadrature_mt].get_submesh_tuples(
            _cell_facet_pairs[quadrature_mt]);

    // Get (cell, local_facet_index) tuples on candidate submesh
    const std::vector<std::int32_t> candidate_facets
        = _submeshes[candidate_mt].get_submesh_tuples(
            _cell_facet_pairs[candidate_mt]);

    auto [candidate_map, reference_x, shape]
        = dolfinx_contact::compute_distance_map(
            *quadrature_mesh, quadrature_facets, *candidate_mesh,
            candidate_facets, *_quadrature_rule, _mode);

    // Compute values of basis functions for all y = Pi(x) in qp
    auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace>(
        _submeshes[candidate_mt].create_functionspace(_V));
    dolfinx::fem::Function<PetscScalar> u_sub(V_sub);
    std::shared_ptr<const dolfinx::fem::DofMap> sub_dofmap = V_sub->dofmap();
    assert(sub_dofmap);
    const int bs_dof = sub_dofmap->bs();
    _submeshes[candidate_mt].copy_function(*u, u_sub);

    std::shared_ptr<const dolfinx::fem::FiniteElement> element
        = V_sub->element();
    std::array<std::size_t, 4> b_shape
        = element->basix_element().tabulate_shape(0, shape[0]);
    if (b_shape.back() > 1)
      throw std::invalid_argument("pack_test_functions assumes values size 1");
    std::vector<double> basis_valuesb(
        std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
    element->tabulate(basis_valuesb, reference_x, shape, 0);

    // Need to apply push forward and dof transformations to test functions
    assert((b_shape.front() == 1) and (b_shape.back() == 1));

    const basix::FiniteElement& b_el = element->basix_element();
    if (element->needs_dof_transformations()
        or b_el.map_type() != basix::maps::type::identity)
    {
      // If we want to do this we need to apply transformation and push
      // forward
      throw std::runtime_error(
          "Packing u on opposite surface functions of space that uses "
          "non-indentity maps is not supported");
    }

    cmdspan4_t basis_values(basis_valuesb.data(), b_shape);
    const std::span<const PetscScalar>& u_coeffs = u_sub.x()->array();

    // Output vector
    const std::size_t num_facets = quadrature_facets.size() / 2;
    const dolfinx::mesh::Topology& topology = candidate_mesh->topology();
    error::check_cell_type(topology.cell_type());
    const std::size_t bs_element = element->block_size();

    const std::size_t num_q_points
        = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
    std::vector<PetscScalar> c(num_facets * num_q_points * bs_element, 0.0);

    // Get cell index on sub-mesh
    const int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    const std::vector<std::int32_t>& facets = candidate_map.array();
    std::vector<std::int32_t> cells(facets.size(), -1);
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
      if (facets[i] < 0)
        continue;
      auto f_cells = f_to_c->links(facets[i]);
      assert(f_cells.size() == 1);
      cells[i] = f_cells.front();
    }

    // Create work vector for expansion coefficients
    const auto cstride = int(num_q_points * bs_element);
    const std::size_t num_basis_functions = b_shape[2];
    const std::size_t value_size = b_shape[3];
    std::vector<PetscScalar> coefficients(num_basis_functions * bs_element);

    for (std::size_t i = 0; i < num_facets; ++i)
    {
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        // Get degrees of freedom for current cell
        if (facets[i * num_q_points + q] < 0)
          continue;
        auto dofs = sub_dofmap->cell_dofs(cells[i * num_q_points + q]);
        for (std::size_t j = 0; j < dofs.size(); ++j)
          for (int k = 0; k < bs_dof; ++k)
            coefficients[bs_dof * j + k] = u_coeffs[bs_dof * dofs[j] + k];

        // Compute expansion
        for (std::size_t k = 0; k < bs_element; ++k)
        {
          for (std::size_t l = 0; l < num_basis_functions; ++l)
          {
            for (std::size_t m = 0; m < value_size; ++m)
            {
              c[cstride * i + q * bs_element + k]
                  += coefficients[bs_element * l + k]
                     * basis_values(0, num_q_points * i + q, l, m);
            }
          }
        }
      }
    }
    t.stop();
    return {std::move(c), cstride};
  }

  /// Compute gradient of function on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[in] u_packed -u packed on opposite surface per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_grad_u_contact(int pair,
                      std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
                      const std::span<const PetscScalar> gap,
                      const std::span<const PetscScalar> u_packed);

  /// Compute outward surface normal at x
  /// @param[in] pair - index of contact pair
  /// @returns c - (normals, cstride) ny packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_nx(int pair);

  /// Compute inward surface normal at Pi(x)
  /// @param[in] pair - index of contact pair
  /// @returns c - normals ny packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_ny(int pair);

  /// Pack gap with rigid surface defined by x[gdim-1] = -g.
  /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
  /// Gap = x - g_vec
  /// @param[in] pair - index of contact pair
  /// @param[in] g - defines location of plane
  /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the
  /// jth component of the Gap on the ith facet at kth quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int pair, double g)
  {
    int quadrature_mt = _contact_pairs[pair][0];
    // Mesh info
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
    assert(mesh);

    const int gdim = mesh->geometry().dim(); // geometrical dimension

    // Tabulate basis function on reference cell (_phi_ref_facets)
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
    std::tie(_reference_basis, _reference_shape)
        = impl::tabulate(cmap, _quadrature_rule);

    // Compute quadrature points on physical facet _qp_phys_"quadrature_mt"
    create_q_phys(quadrature_mt);
    const std::size_t num_facets = _cell_facet_pairs[quadrature_mt].size() / 2;
    // FIXME: This does not work for prism meshes
    std::size_t num_q_point
        = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
    mdspan3_t qp_span(_qp_phys[quadrature_mt].data(), num_facets, num_q_point,
                      gdim);
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = (int)num_q_point * gdim;
    for (std::size_t i = 0; i < num_facets; i++)
    {
      int offset = (int)i * cstride;
      for (std::size_t k = 0; k < num_q_point; k++)
        c[offset + (k + 1) * gdim - 1] = g - qp_span(i, k, gdim - 1);
    }
    return {std::move(c), cstride};
  }
  /// This function updates the submesh geometry for all submeshes using
  /// a function given on the parent mesh
  /// @param[in] u - displacement
  void update_submesh_geometry(dolfinx::fem::Function<PetscScalar>& u) const;

private:
  std::shared_ptr<QuadratureRule> _quadrature_rule; // quadrature rule
  std::vector<int> _surfaces; // meshtag values for surfaces
  // store index of candidate_surface for each puppet_surface
  std::vector<std::array<int, 2>> _contact_pairs;
  std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space
  // _facets_maps[i] = adjacency list of closest facet on candidate surface
  // for every quadrature point in _qp_phys[i] (quadrature points on every
  // facet of ith surface)
  std::vector<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>>
      _facet_maps;
  //  _qp_phys[i] contains the quadrature points on the physical facets for
  //  each facet on ith surface in _surfaces
  std::vector<std::vector<double>> _qp_phys;
  // quadrature points on facets of reference cell
  std::vector<double> _reference_basis;
  std::array<std::size_t, 4> _reference_shape;
  // maximum number of cells linked to a cell on ith surface
  std::vector<std::size_t> _max_links;
  // submeshes for contact surface
  std::vector<SubMesh> _submeshes;
  // facets as (cell, facet) pairs. The pairs are flattened row-major
  std::vector<std::vector<std::int32_t>> _cell_facet_pairs;

  // Contact search mode
  ContactMode _mode;
};
} // namespace dolfinx_contact
