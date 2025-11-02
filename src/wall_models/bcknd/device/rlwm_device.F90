! Copyright (c) 2025, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
!   * Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.
!
!   * Redistributions in binary form must reproduce the above
!     copyright notice, this list of conditions and the following
!     disclaimer in the documentation and/or other materials provided
!     with the distribution.
!
!   * Neither the name of the authors nor the names of its
!     contributors may be used to endorse or promote products derived
!     from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!> Implements the device kernel for the `rlwm_t` type.
module rlwm_device
  use num_types, only : rp, c_rp
  use, intrinsic :: iso_c_binding, only : c_ptr
  use utils, only : neko_error
  implicit none
  private

#ifdef HAVE_HIP
  interface
     subroutine hip_rlwm_compute(u_d, v_d, w_d, &
          ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
          n_x_d, n_y_d, n_z_d, nu_d, h_d, &
          tau_x_d, tau_y_d, tau_z_d, n_nodes, lx, &
          kappa, B, tstep) &
          bind(c, name = 'hip_rlwm_compute')
       use, intrinsic :: iso_c_binding, only : c_ptr, c_int
       use num_types, only : c_rp
       implicit none
       type(c_ptr), value :: u_d, v_d, w_d
       type(c_ptr), value :: ind_r_d, ind_s_d, ind_t_d, ind_e_d
       type(c_ptr), value :: n_x_d, n_y_d, n_z_d, h_d, nu_d
       real(c_rp) :: kappa, B
       type(c_ptr), value :: tau_x_d, tau_y_d, tau_z_d
       integer(c_int) :: n_nodes, lx, tstep
     end subroutine hip_rlwm_compute
  end interface
  
#elif HAVE_CUDA

  interface
    subroutine cuda_spalding_initialize(u_d, v_d, w_d, &
      ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
      n_x_d, n_y_d, n_z_d, nu_d, h_d, &
      tau_x_d, tau_y_d, tau_z_d, n_nodes, lx, &
      kappa, B, tstep, &
      tau_true, &
      ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
      tau_old_l_d, tau_new_l_d, &
      l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
      error_new_d, error_old_d, rel_error_d, &
      reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
      terminal_d, &
      state_d, action_d, &
      msk_d, reward_field_d, slope_field_d, intercept_field_d) &
      bind(c, name = 'cuda_spalding_initialize')

      use, intrinsic :: iso_c_binding, only : c_ptr, c_int
      use num_types, only : c_rp
      implicit none

      type(c_ptr), value :: u_d, v_d, w_d
      type(c_ptr), value :: ind_r_d, ind_s_d, ind_t_d, ind_e_d
      type(c_ptr), value :: n_x_d, n_y_d, n_z_d, h_d, nu_d
      real(c_rp) :: kappa, B
      type(c_ptr), value :: tau_x_d, tau_y_d, tau_z_d
      integer(c_int) :: n_nodes, lx, tstep

      ! TorchFort
      real(c_rp) :: tau_true
      type(c_ptr), value :: ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d
      type(c_ptr), value :: tau_old_l_d, tau_new_l_d
      type(c_ptr), value :: l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d
      type(c_ptr), value :: error_new_d, error_old_d, rel_error_d
      type(c_ptr), value :: reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d
      type(c_ptr), value :: terminal_d
      type(c_ptr), value :: state_d, action_d
      type(c_ptr), value :: msk_d, reward_field_d, slope_field_d, intercept_field_d

    end subroutine cuda_spalding_initialize
  end interface

  interface
    subroutine cuda_rlwm_compute(u_d, v_d, w_d, &
      ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
      n_x_d, n_y_d, n_z_d, nu_d, h_d, &
      tau_x_d, tau_y_d, tau_z_d, n_nodes, lx, &
      kappa, B, tstep, &
      model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs, tau_true, &
      ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
      tau_old_l_d, tau_new_l_d, &
      l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
      error_new_d, error_old_d, rel_error_d, &
      reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
      terminal_d, &
      state_d, action_d, &
      msk_d, reward_field_d, slope_field_d, intercept_field_d) &
      bind(c, name = 'cuda_rlwm_compute')

      use, intrinsic :: iso_c_binding, only : c_ptr, c_int
      use num_types, only : c_rp
      implicit none

      type(c_ptr), value :: u_d, v_d, w_d
      type(c_ptr), value :: ind_r_d, ind_s_d, ind_t_d, ind_e_d
      type(c_ptr), value :: n_x_d, n_y_d, n_z_d, h_d, nu_d
      real(c_rp) :: kappa, B
      type(c_ptr), value :: tau_x_d, tau_y_d, tau_z_d
      integer(c_int) :: n_nodes, lx, tstep

      ! TorchFort
      integer(c_int) :: model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs
      real(c_rp) :: tau_true
      type(c_ptr), value :: ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d
      type(c_ptr), value :: tau_old_l_d, tau_new_l_d
      type(c_ptr), value :: l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d
      type(c_ptr), value :: error_new_d, error_old_d, rel_error_d
      type(c_ptr), value :: reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d
      type(c_ptr), value :: terminal_d
      type(c_ptr), value :: state_d, action_d
      type(c_ptr), value :: msk_d, reward_field_d, slope_field_d, intercept_field_d

    end subroutine cuda_rlwm_compute
  end interface

  interface
    subroutine cuda_rlwm_actuate(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
      action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
      tau_x_d, tau_y_d, tau_z_d, tau_true, &
      ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
      error_new_d, error_old_d, rel_error_d, &
      reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
      msk_d, reward_field_d) &
      bind(c, name = 'cuda_rlwm_actuate')

      use, intrinsic :: iso_c_binding, only : c_ptr, c_int
      use num_types, only : c_rp
      implicit none

      integer(c_int) :: n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length
      real(c_rp) :: tau_true
      type(c_ptr), value :: action_d
      type(c_ptr), value :: tau_old_l_d, tau_new_l_d, utau_l_d
      type(c_ptr), value :: tau_x_d, tau_y_d, tau_z_d
      type(c_ptr), value :: ui_l_d, vi_l_d, wi_l_d, magu_l_d
      type(c_ptr), value :: error_new_d, error_old_d, rel_error_d
      type(c_ptr), value :: reward_d, total_reward_d
      type(c_ptr), value :: base_reward_d, bonus_reward_d
      type(c_ptr), value :: msk_d, reward_field_d

    end subroutine cuda_rlwm_actuate
  end interface

  interface
    subroutine cuda_rlwm_under_relax(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
      action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
      tau_x_d, tau_y_d, tau_z_d, tau_true, &
      ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
      error_new_d, error_old_d, rel_error_d, &
      reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
      msk_d, reward_field_d) &
      bind(c, name = 'cuda_rlwm_under_relax')

      use, intrinsic :: iso_c_binding, only : c_ptr, c_int
      use num_types, only : c_rp
      implicit none

      integer(c_int) :: n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length
      real(c_rp) :: tau_true
      type(c_ptr), value :: action_d
      type(c_ptr), value :: tau_old_l_d, tau_new_l_d, utau_l_d
      type(c_ptr), value :: tau_x_d, tau_y_d, tau_z_d
      type(c_ptr), value :: ui_l_d, vi_l_d, wi_l_d, magu_l_d
      type(c_ptr), value :: error_new_d, error_old_d, rel_error_d
      type(c_ptr), value :: reward_d, total_reward_d
      type(c_ptr), value :: base_reward_d, bonus_reward_d
      type(c_ptr), value :: msk_d, reward_field_d

    end subroutine cuda_rlwm_under_relax
  end interface
  
#elif HAVE_OPENCL
#endif
  public :: spalding_initialize_device, rlwm_compute_device, rlwm_actuate_device, rlwm_under_relax_device

contains
  !> Compute the wall shear stress on device using RLWM.
  !! @param t The time value.
  !! @param tstep The current time-step.
  subroutine rlwm_compute_device(u_d, v_d, w_d, &
    ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
    n_x_d, n_y_d, n_z_d, nu_d, h_d, tau_x_d, tau_y_d, tau_z_d, &
    n_nodes, lx, kappa, B, tstep, &
    model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs, tau_true, &
    ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
    tau_old_l_d, tau_new_l_d, &
    l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
    error_new_d, error_old_d, rel_error_d, &
    reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
    terminal_d, &
    state_d, action_d, &
    msk_d, reward_field_d, slope_field_d, intercept_field_d)

    integer, intent(in) :: n_nodes, lx, tstep
    type(c_ptr), intent(in) :: u_d, v_d, w_d
    type(c_ptr), intent(in) :: ind_r_d, ind_s_d, ind_t_d, ind_e_d
    type(c_ptr), intent(in) :: n_x_d, n_y_d, n_z_d, h_d, nu_d
    type(c_ptr), intent(inout) :: tau_x_d, tau_y_d, tau_z_d
    real(kind=rp), intent(in) :: kappa, B
    ! TorchFort
    integer, intent(in) :: model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs
    real(kind=rp), intent(in) :: tau_true
    type(c_ptr), intent(inout) :: ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d
    type(c_ptr), intent(inout) :: tau_old_l_d, tau_new_l_d
    type(c_ptr), intent(inout) :: l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d
    type(c_ptr), intent(inout) :: error_new_d, error_old_d, rel_error_d
    type(c_ptr), intent(inout) :: reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d
    type(c_ptr), intent(inout) :: terminal_d
    type(c_ptr), intent(inout) :: state_d, action_d
    type(c_ptr), intent(inout) :: msk_d, reward_field_d, slope_field_d, intercept_field_d

#if HAVE_HIP
    call neko_error("HIP is not implemented for RLWM")
#elif HAVE_CUDA
    call cuda_rlwm_compute(u_d, v_d, w_d, &
         ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
         n_x_d, n_y_d, n_z_d, nu_d, h_d, &
         tau_x_d, tau_y_d, tau_z_d, n_nodes, lx, kappa, B, tstep, &
         model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs, tau_true, &
         ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
         tau_old_l_d, tau_new_l_d, &
         l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
         error_new_d, error_old_d, rel_error_d, &
         reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
         terminal_d, &
         state_d, action_d, &
         msk_d, reward_field_d, slope_field_d, intercept_field_d)
#elif HAVE_OPENCL
    call neko_error("OPENCL is not implemented for RLWM")
#else
    call neko_error('No device backend configured')
#endif

  end subroutine rlwm_compute_device

  !> Compute the wall shear stress on device using Spalding's law
  !! @param t The time value.
  !! @param tstep The current time-step.
  subroutine spalding_initialize_device(u_d, v_d, w_d, &
    ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
    n_x_d, n_y_d, n_z_d, nu_d, h_d, tau_x_d, tau_y_d, tau_z_d, &
    n_nodes, lx, kappa, B, tstep, &
    tau_true, &
    ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
    tau_old_l_d, tau_new_l_d, &
    l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
    error_new_d, error_old_d, rel_error_d, &
    reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
    terminal_d, &
    state_d, action_d, &
    msk_d, reward_field_d, slope_field_d, intercept_field_d)

    integer, intent(in) :: n_nodes, lx, tstep
    type(c_ptr), intent(in) :: u_d, v_d, w_d
    type(c_ptr), intent(in) :: ind_r_d, ind_s_d, ind_t_d, ind_e_d
    type(c_ptr), intent(in) :: n_x_d, n_y_d, n_z_d, h_d, nu_d
    type(c_ptr), intent(inout) :: tau_x_d, tau_y_d, tau_z_d
    real(kind=rp), intent(in) :: kappa, B
    ! TorchFort
    real(kind=rp), intent(in) :: tau_true
    type(c_ptr), intent(inout) :: ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d
    type(c_ptr), intent(inout) :: tau_old_l_d, tau_new_l_d
    type(c_ptr), intent(inout) :: l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d
    type(c_ptr), intent(inout) :: error_new_d, error_old_d, rel_error_d
    type(c_ptr), intent(inout) :: reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d
    type(c_ptr), intent(inout) :: terminal_d
    type(c_ptr), intent(inout) :: state_d, action_d
    type(c_ptr), intent(inout) :: msk_d, reward_field_d, slope_field_d, intercept_field_d

#if HAVE_HIP
    call neko_error("HIP is not implemented for Spalding")
#elif HAVE_CUDA
    call cuda_spalding_initialize(u_d, v_d, w_d, &
         ind_r_d, ind_s_d, ind_t_d, ind_e_d, &
         n_x_d, n_y_d, n_z_d, nu_d, h_d, &
         tau_x_d, tau_y_d, tau_z_d, n_nodes, lx, kappa, B, tstep, &
         tau_true, &
         ui_l_d, vi_l_d, wi_l_d, normu_l_d, magu_l_d, vg_l_d, utau_l_d, &
         tau_old_l_d, tau_new_l_d, &
         l_star_d, u_plus_d, g_plus_d, h_plus_d, slope_d, intercept_d, dudy_d, &
         error_new_d, error_old_d, rel_error_d, &
         reward_d, total_reward_d, reward_out_d, base_reward_d, bonus_reward_d, &
         terminal_d, &
         state_d, action_d, &
         msk_d, reward_field_d, slope_field_d, intercept_field_d)
#elif HAVE_OPENCL
    call neko_error("OPENCL is not implemented for Spalding")
#else
    call neko_error('No device backend configured')
#endif

  end subroutine spalding_initialize_device

  !> Apply ML model actions to wall shear stress on device using RLWM.
  !! @param n_nodes Number of wall nodes.
  !! @param tstep Current time step.
  !! @param tsteps_rl Number of time steps between RL actions.
  subroutine rlwm_actuate_device(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
    action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
    tau_x_d, tau_y_d, tau_z_d, tau_true, &
    ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
    error_new_d, error_old_d, rel_error_d, &
    reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
    msk_d, reward_field_d)

    integer, intent(in) :: n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length
    real(kind=rp), intent(in) :: tau_true
    type(c_ptr), intent(in) :: action_d
    type(c_ptr), intent(inout) :: tau_old_l_d, tau_new_l_d, utau_l_d
    type(c_ptr), intent(inout) :: tau_x_d, tau_y_d, tau_z_d
    type(c_ptr), intent(inout) :: ui_l_d, vi_l_d, wi_l_d, magu_l_d
    type(c_ptr), intent(inout) :: error_new_d, error_old_d, rel_error_d
    type(c_ptr), intent(inout) :: reward_d, total_reward_d
    type(c_ptr), intent(inout) :: base_reward_d, bonus_reward_d
    type(c_ptr), intent(inout) :: msk_d, reward_field_d

#if HAVE_HIP
    call neko_error("HIP is not implemented for RLWM actuate")
#elif HAVE_CUDA
    call cuda_rlwm_actuate(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
         action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
         tau_x_d, tau_y_d, tau_z_d, tau_true, &
         ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
         error_new_d, error_old_d, rel_error_d, &
         reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
         msk_d, reward_field_d)
#elif HAVE_OPENCL
    call neko_error("OPENCL is not implemented for RLWM actuate")
#else
    call neko_error('No device backend configured')
#endif

  end subroutine rlwm_actuate_device

  !> Apply under relaxation to wall shear stress on device using RLWM.
  !! @param n_nodes Number of wall nodes.
  !! @param tstep Current time step.
  !! @param tsteps_rl Number of time steps between RL actions.
  subroutine rlwm_under_relax_device(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
    action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
    tau_x_d, tau_y_d, tau_z_d, tau_true, &
    ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
    error_new_d, error_old_d, rel_error_d, &
    reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
    msk_d, reward_field_d)

    integer, intent(in) :: n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length
    real(kind=rp), intent(in) :: tau_true
    type(c_ptr), intent(in) :: action_d
    type(c_ptr), intent(inout) :: tau_old_l_d, tau_new_l_d, utau_l_d
    type(c_ptr), intent(inout) :: tau_x_d, tau_y_d, tau_z_d
    type(c_ptr), intent(inout) :: ui_l_d, vi_l_d, wi_l_d, magu_l_d
    type(c_ptr), intent(inout) :: error_new_d, error_old_d, rel_error_d
    type(c_ptr), intent(inout) :: reward_d, total_reward_d
    type(c_ptr), intent(inout) :: base_reward_d, bonus_reward_d
    type(c_ptr), intent(inout) :: msk_d, reward_field_d

#if HAVE_HIP
    call neko_error("HIP is not implemented for RLWM under_relax")
#elif HAVE_CUDA
    call cuda_rlwm_under_relax(n_nodes, tstep, start_rl_tstep, tsteps_rl, episode_length, &
         action_d, tau_old_l_d, tau_new_l_d, utau_l_d, &
         tau_x_d, tau_y_d, tau_z_d, tau_true, &
         ui_l_d, vi_l_d, wi_l_d, magu_l_d, &
         error_new_d, error_old_d, rel_error_d, &
         reward_d, total_reward_d, base_reward_d, bonus_reward_d, &
         msk_d, reward_field_d)
#elif HAVE_OPENCL
    call neko_error("OPENCL is not implemented for RLWM under_relax")
#else
    call neko_error('No device backend configured')
#endif

  end subroutine rlwm_under_relax_device


end module rlwm_device