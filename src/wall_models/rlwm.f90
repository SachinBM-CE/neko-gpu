! Copyright (c) 2024, The Neko Authors
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
!
!> Implements `rlwm_t`.
module rlwm
  use field, only: field_t
  use num_types, only : rp
  use json_module, only : json_file
  use coefs, only : coef_t
  use neko_config, only : NEKO_BCKND_DEVICE
  use wall_model, only : wall_model_t
  use field_registry, only : neko_field_registry
  use json_utils, only : json_get_or_default
  use rlwm_cpu, only : rlwm_compute_cpu
  use rlwm_device, only : spalding_initialize_device, rlwm_compute_device, rlwm_actuate_device
  use field_math, only: field_invcol3
  use vector, only : vector_t
  use math, only: masked_gather_copy_0
  use device_math, only: device_masked_gather_copy_0
  use scratch_registry, only : neko_scratch_registry
  
  ! ****** TorchFort ******
  use torchfort
  use comm, only : pe_rank, pe_size, NEKO_COMM
  use iso_c_binding
  use math
  use tensor
  use num_types, only : sp
  use mpi_f08, only : MPI_INTEGER, MPI_SUCCESS, MPI_SUM, MPI_Allgather, MPI_Allreduce, &
                      MPI_Gatherv, MPI_Scatterv, MPI_DOUBLE_PRECISION, MPI_Wtime
  use json_utils, only : json_get
  use utils, only : neko_error
  use operators, only : grad, dudxyz
  use device, only: device_map, device_free, device_associated, device_memcpy, DEVICE_TO_HOST, HOST_TO_DEVICE
  use, intrinsic :: iso_c_binding, only : c_ptr

  implicit none
  private

  !> Wall model based on rlwm's law of the wall.
  !! Reference: http://dx.doi.org/10.1115/1.3641728
  type, public, extends(wall_model_t) :: rlwm_t
     !> The von Karman coefficient.
     real(kind=rp) :: kappa = 0.41_rp
     !> The log-law intercept.
     real(kind=rp) :: B = 5.2_rp
     !> The kinematic viscosity.
     type(vector_t) :: nu
	 
    !> TorchFort =================================================================================================================
    !> JSON INPUTS 
    character(len=256) :: tf_key, yaml_path, log_dir, policy_method, phase
    integer :: model_device, rb_device, start_rl_tstep, tsteps_rl, episode_length, n_epochs
    real(kind=rp) :: tau_true
    !> Vectors
    type(vector_t) :: ui_l, vi_l, wi_l, normu_l, magu_l, vg_l, utau_l, tau_old_l, tau_new_l, & 
              l_star, u_plus, g_plus, h_plus, slope, intercept, &
              error_new, error_old, rel_error, &
              reward, total_reward, reward_out, base_reward, bonus_reward, &
              terminal
    !> 4D Arrays 
    real(kind=rp), dimension(:,:,:,:), allocatable :: dudy
    type(c_ptr) :: dudy_d = C_NULL_PTR
    !> 2D Arrays
    real(kind=rp), dimension(:,:), allocatable :: state, action
    type(c_ptr) :: state_d = C_NULL_PTR, action_d = C_NULL_PTR
    !> MPI
    integer :: total_agents, episode=0
    integer, dimension(:), allocatable :: recvcounts, displs
    real(kind=rp), dimension(:), allocatable :: global_reward, global_terminal
    real(kind=rp), dimension(:,:), allocatable :: global_state, global_state_old, global_state_older, & 
                                                  global_action, global_action_old, global_action_older
    !> Fields 
    type(field_t), pointer :: reward_field => null(), slope_field => null(), intercept_field => null()
    !> Single precision loss values
    real(kind=sp) :: p_loss_val, q_loss_val
    !=============================================================================================================================

   contains
     !> Constructor from JSON.
     procedure, pass(this) :: init => rlwm_init
     !> Partial constructor from JSON, meant to work as the first stage of
     !! initialization before the `finalize` call.
     procedure, pass(this) :: partial_init => rlwm_partial_init
     !> Finalize the construction using the mask and facet arrays of the bc.
     procedure, pass(this) :: finalize => rlwm_finalize
     !> Constructor from components.
     procedure, pass(this) :: init_from_components => rlwm_init_from_components
     !> Destructor.
     procedure, pass(this) :: free => rlwm_free
     !> Compute the kinematic viscosity at the wall.
     procedure, pass(this) :: compute_nu => rlwm_compute_nu
     !> Compute the wall shear stress.
     procedure, pass(this) :: compute => rlwm_compute

     !> Newly added subroutines for TorchFort-GPU
     procedure, pass(this) :: get_global_state => rlwm_get_global_state
     procedure, pass(this) :: get_device_action => rlwm_get_device_action
     procedure, pass(this) :: predict_global_action => rlwm_predict_global_action
     procedure, pass(this) :: get_global_reward => rlwm_get_global_reward
     procedure, pass(this) :: get_global_terminal => rlwm_get_global_terminal
     procedure, pass(this) :: update_buffer => rlwm_update_buffer
     procedure, pass(this) :: train_and_save => rlwm_train_and_save

  end type rlwm_t

contains

  !> Constructor from JSON.
  !! @param scheme_name The name of the scheme for which the wall model is used.
  !! @param coef SEM coefficients.
  !! @param msk The boundary mask.
  !! @param facet The boundary facets.
  !! @param h_index The off-wall index of the sampling cell.
  !! @param json A dictionary with parameters.
  subroutine rlwm_init(this, scheme_name, coef, msk, facet, h_index, json)
    class(rlwm_t), intent(inout) :: this
    character(len=*), intent(in) :: scheme_name
    type(coef_t), intent(in) :: coef
    integer, intent(in) :: msk(:)
    integer, intent(in) :: facet(:)
    integer, intent(in) :: h_index
    type(json_file), intent(inout) :: json
    real(kind=rp) :: kappa, B
	
    ! ****** TorchFort ****** 
    integer :: res, ierr, i
    ! ***********************
    
    call json_get_or_default(json, "kappa", kappa, 0.41_rp)
    call json_get_or_default(json, "B", B, 5.2_rp)
    
    call this%init_from_components(scheme_name, coef, msk, facet, h_index, kappa, B)

  end subroutine rlwm_init

  !> Constructor from JSON.
  !! @param coef SEM coefficients.
  !! @param json A dictionary with parameters.
  subroutine rlwm_partial_init(this, coef, json)
    class(rlwm_t), intent(inout) :: this
    type(coef_t), intent(in) :: coef
    type(json_file), intent(inout) :: json
	
    ! ****** TorchFort ****** 
    integer :: res, ierr, i
    real(kind=rp) :: tmp_real
    character(len=:), allocatable :: tmp_string
    ! ***********************

    call this%partial_init_base(coef, json)
    call json_get_or_default(json, "kappa", this%kappa, 0.41_rp)
    call json_get_or_default(json, "B", this%B, 5.2_rp)

    print *, "rlwm_partial_init called"

    call json_get(json, "tf_key", tmp_string)
    this%tf_key = trim(tmp_string)

    call json_get(json, "yaml_path", tmp_string)
    this%yaml_path = trim(tmp_string)

    call json_get(json, "log_dir", tmp_string)
    this%log_dir = trim(tmp_string)
    
    call json_get(json, "policy_method", tmp_string)
    this%policy_method = trim(tmp_string)

    call json_get(json, "phase", tmp_string)
    this%phase = trim(tmp_string)

    call json_get_or_default(json, "model_device", tmp_real, -1.0_rp)
    this%model_device = int(tmp_real)

    call json_get_or_default(json, "rb_device", tmp_real, -1.0_rp)
    this%rb_device = int(tmp_real)

    call json_get_or_default(json, "tau_true", tmp_real, 0.002_rp)
    this%tau_true = tmp_real
    
    call json_get_or_default(json, "start_rl_tstep", tmp_real, 1000.0_rp)
    this%start_rl_tstep = int(tmp_real)
    
    call json_get_or_default(json, "tsteps_rl", tmp_real, 100.0_rp)
    this%tsteps_rl = int(tmp_real)

    call json_get_or_default(json, "episode_length", tmp_real, 100.0_rp)
    this%episode_length = int(tmp_real)

    call json_get_or_default(json, "n_epochs", tmp_real, 10.0_rp)
    this%n_epochs = int(tmp_real)
    
    res = torchfort_set_manual_seed(123)
    if (res /= TORCHFORT_RESULT_SUCCESS) stop
    print *, "Result of set_manual_seed : ", res
    print *
    
    if (pe_rank .eq. 0) then
      select case (trim(this%policy_method))
      case ("on-policy")
        res = torchfort_rl_on_policy_create_system(this%tf_key, this%yaml_path, this%model_device, this%rb_device)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
        print *, "Result of on_policy_create_system : ", res
        print *
      case ("off-policy")
        res = torchfort_rl_off_policy_create_system(this%tf_key, this%yaml_path, this%model_device, this%rb_device)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
        print *, "Result of off_policy_create_system : ", res
        ! res = torchfort_rl_off_policy_create_distributed_system(this%tf_key, & 
        ! this%yaml_path, NEKO_COMM, this%model_device, this%rb_device)
        ! if (res /= TORCHFORT_RESULT_SUCCESS) stop
        ! print *, "Result of create_distributed_system : ", res
        print *
      case default
        print *, "Unknown command: ", trim(this%policy_method)
        stop 1
      end select
    end if
    
    if (trim(this%phase) .eq. 'testing') then
      res = torchfort_rl_off_policy_load_checkpoint(this%tf_key, this%log_dir)
      print *, "Result of load_checkpoint : ", res
      if (res /= TORCHFORT_RESULT_SUCCESS) stop
    end if
    
    ! print *, "===> pe_size :", pe_size	
    ! print *, "this%n_nodes : ", this%n_nodes, "from pe_rank: ", pe_rank
    ! print *, "this%msk(0) : ", this%msk(0), "from pe_rank: ", pe_rank
    
    allocate(this%dudy(coef%Xh%lx, coef%Xh%ly, coef%Xh%lz, coef%msh%nelv))

    if (NEKO_BCKND_DEVICE .eq. 1) then
      call device_map(this%dudy, this%dudy_d, size(this%dudy))
    end if

  end subroutine rlwm_partial_init

  !> Finalize the construction using the mask and facet arrays of the bc.
  !! @param msk The boundary mask.
  !! @param facet The boundary facets.
  subroutine rlwm_finalize(this, msk, facet)
    class(rlwm_t), intent(inout) :: this
    integer, intent(in) :: msk(:)
    integer, intent(in) :: facet(:)
	
	  ! ****** TorchFort ****** 
	  integer :: res, ierr, i
	  ! ***********************

    call this%finalize_base(msk, facet)
    call this%nu%init(this%n_nodes)
    
    !> Vectors
    call this%ui_l%init(this%n_nodes)
    call this%vi_l%init(this%n_nodes)
    call this%wi_l%init(this%n_nodes)
    call this%normu_l%init(this%n_nodes)
    call this%magu_l%init(this%n_nodes)
    call this%vg_l%init(this%n_nodes)
    call this%utau_l%init(this%n_nodes)
    call this%tau_old_l%init(this%n_nodes)
    call this%tau_new_l%init(this%n_nodes)
    
    call this%l_star%init(this%n_nodes)
    call this%u_plus%init(this%n_nodes)
    call this%g_plus%init(this%n_nodes)
    call this%h_plus%init(this%n_nodes)
    call this%slope%init(this%n_nodes)
    call this%intercept%init(this%n_nodes)
    
    call this%error_new%init(this%n_nodes)
    call this%error_old%init(this%n_nodes)
    call this%rel_error%init(this%n_nodes)
    
    call this%reward%init(this%n_nodes)
    call this%total_reward%init(this%n_nodes)
    call this%reward_out%init(this%n_nodes)
    call this%base_reward%init(this%n_nodes)
    call this%bonus_reward%init(this%n_nodes)
    
    call this%terminal%init(this%n_nodes)

    if (this%n_nodes > 0) then
        allocate(this%state(2, this%n_nodes), this%action(1, this%n_nodes))
    else
        allocate(this%state(2, 1), this%action(1, 1))
    end if

    if (NEKO_BCKND_DEVICE .eq. 1) then
      call device_map(this%state, this%state_d, size(this%state))
      call device_map(this%action, this%action_d, size(this%action))
    end if
    
    !> Collecting receive counts, displacements & total agents ====================================================================
    
    allocate(this%recvcounts(0:(pe_size-1)), this%displs(0:(pe_size-1)))
    
    ! Gather the number of agents (this%n_nodes) from all ranks onto all ranks
    call MPI_Allgather(this%n_nodes, 1, MPI_INTEGER, this%recvcounts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    if (ierr /= MPI_SUCCESS) then
      call neko_error("MPI_Allgather failed in rlwm_finalize")
    end if
    print *, "recvcounts = ", this%recvcounts
    
    ! Calculate the displacements for MPI_Gatherv (all ranks need this)
    this%displs(0) = 0
    do i = 1, (pe_size - 1)
      this%displs(i) = this%displs(i-1) + this%recvcounts(i-1)
    end do
    print *, "displs = ", this%displs
    
    ! Get the total number of agents across all ranks
    call MPI_Allreduce(this%n_nodes, this%total_agents, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (ierr /= MPI_SUCCESS) then
      call neko_error("MPI_Allreduce failed in rlwm_finalize")
    end if
    print *, ">>>> total_agents = ", this%total_agents
    
    !==============================================================================================================================
    
    ! Allocate global arrays
    if (pe_rank == 0) then
      allocate(this%global_state(2,this%total_agents))
      allocate(this%global_state_old(2,this%total_agents))
      allocate(this%global_state_older(2,this%total_agents))
      allocate(this%global_action(1,this%total_agents))
      allocate(this%global_action_old(1,this%total_agents))
      allocate(this%global_action_older(1,this%total_agents))
      allocate(this%global_reward(this%total_agents))
      allocate(this%global_terminal(this%total_agents))
    else
      ! For non-root processes, these can be unallocated or size 1
      allocate(this%global_state(1,1))
      allocate(this%global_state_old(1,1))
      allocate(this%global_state_older(1,1))
      allocate(this%global_action(1,1))
      allocate(this%global_action_old(1,1))
      allocate(this%global_action_older(1,1))
      allocate(this%global_reward(1))
      allocate(this%global_terminal(1))
    end if
    
    ! Adding fields to field registry
    call neko_field_registry%add_field(this%dof, "reward", ignore_existing = .true.)
    this%reward_field => neko_field_registry%get_field("reward")
    call neko_field_registry%add_field(this%dof, "slope", ignore_existing = .true.)
    this%slope_field => neko_field_registry%get_field("slope")
    call neko_field_registry%add_field(this%dof, "intercept", ignore_existing = .true.)
    this%intercept_field => neko_field_registry%get_field("intercept")

    print *, "**********************************"
    print *, "|||| Initialization Completed ||||"
    print *, "**********************************"
    print *	
	
  end subroutine rlwm_finalize

  !> Constructor from components.
  !! @param scheme_name The name of the scheme for which the wall model is used.
  !! @param coef SEM coefficients.
  !! @param msk The boundary mask.
  !! @param facet The boundary facets.
  !! @param h_index The off-wall index of the sampling cell.
  !! @param kappa The von Karman coefficient.
  !! @param B The log-law intercept.
  subroutine rlwm_init_from_components(this, scheme_name, coef, msk, &
       facet, h_index, kappa, B)
    class(rlwm_t), intent(inout) :: this
    character(len=*), intent(in) :: scheme_name
    type(coef_t), intent(in) :: coef
    integer, intent(in) :: msk(:)
    integer, intent(in) :: facet(:)
    integer, intent(in) :: h_index
    real(kind=rp), intent(in) :: kappa
    real(kind=rp), intent(in) :: B

    call this%free()
    call this%init_base(scheme_name, coef, msk, facet, h_index)

    this%kappa = kappa
    this%B = B

    call this%nu%init(this%n_nodes)
  end subroutine rlwm_init_from_components

  !> Compute the kinematic viscosity vector.
  subroutine rlwm_compute_nu(this)
    class(rlwm_t), intent(inout) :: this
    type(field_t), pointer :: temp
    integer :: idx

    call neko_scratch_registry%request_field(temp, idx)
    call field_invcol3(temp, this%mu, this%rho)

    if (NEKO_BCKND_DEVICE .eq. 1) then
       call device_masked_gather_copy_0(this%nu%x_d, temp%x_d, this%msk_d, &
            temp%size(), this%nu%size())
    else
       call masked_gather_copy_0(this%nu%x, temp%x, this%msk, temp%size(), &
            this%nu%size())
    end if

    call neko_scratch_registry%relinquish_field(idx)
  end subroutine rlwm_compute_nu

  !> Destructor for the rlwm_t (base) class.
  subroutine rlwm_free(this)
    class(rlwm_t), intent(inout) :: this

    call this%free_base()
	
    if (allocated(this%dudy)) deallocate(this%dudy)
    if (allocated(this%state)) deallocate(this%state)
    if (allocated(this%action)) deallocate(this%action)	

    if (c_associated(this%dudy_d)) then
        call device_free(this%dudy_d)
    end if
    if (c_associated(this%state_d)) then
        call device_free(this%state_d)
    end if
    if (c_associated(this%action_d)) then
        call device_free(this%action_d)
    end if
    
    !> Vectors
    call this%ui_l%free()
    call this%vi_l%free()
    call this%wi_l%free()
    call this%normu_l%free()
    call this%magu_l%free()
    call this%vg_l%free()
    call this%utau_l%free()
    call this%tau_old_l%free()
    call this%tau_new_l%free()
    
    call this%l_star%free()
    call this%u_plus%free()
    call this%g_plus%free()
    call this%h_plus%free()
    call this%slope%free()
    call this%intercept%free()
    
    call this%error_new%free()
    call this%error_old%free()
    call this%rel_error%free()
    
    call this%reward%free()
    call this%total_reward%free()
    call this%reward_out%free()
    call this%base_reward%free()
    call this%bonus_reward%free()
    
    if (allocated(this%recvcounts)) deallocate(this%recvcounts)
    if (allocated(this%displs)) deallocate(this%displs)
    
    if (allocated(this%global_state)) deallocate(this%global_state)
    if (allocated(this%global_state_old)) deallocate(this%global_state_old)
    if (allocated(this%global_state_older)) deallocate(this%global_state_older)
    if (allocated(this%global_action)) deallocate(this%global_action)
    if (allocated(this%global_action_old)) deallocate(this%global_action_old)
    if (allocated(this%global_action_older)) deallocate(this%global_action_older)
    if (allocated(this%global_reward)) deallocate(this%global_reward)
    if (allocated(this%global_terminal)) deallocate(this%global_terminal)
    
    nullify(this%reward_field)
    nullify(this%slope_field)
    nullify(this%intercept_field)

  end subroutine rlwm_free

  !> Compute the wall shear stress.
  !! @param t The time value.
  !! @param tstep The current time-step.
  subroutine rlwm_compute(this, t, tstep)
    class(rlwm_t), intent(inout) :: this
    real(kind=rp), intent(in) :: t
    integer, intent(in) :: tstep
    type(field_t), pointer :: u
    type(field_t), pointer :: v
    type(field_t), pointer :: w
    integer :: i
    real(kind=rp) :: ui, vi, wi, magu, utau, normu, guess
    ! TorchFort
    integer :: res, ierr, epoch
    logical :: is_ready
    real(kind=rp) :: start_time, end_time, train_starter

    ! ::::> Pre-computation setup ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    ! start_time = MPI_WTIME()
    call this%compute_nu()
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: Compute Nu = ', (end_time - start_time) * 1000.0_rp, ' ms'

    u => neko_field_registry%get_field("u")
    v => neko_field_registry%get_field("v")
    w => neko_field_registry%get_field("w")
	
	  ! Gradient Tensor
    ! start_time = MPI_WTIME()
	  call dudxyz(this%dudy, u%x, this%coef%drdy, this%coef%dsdy, this%coef%dtdy, this%coef)
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: Gradient Tensor (dudxyz) = ', (end_time - start_time) * 1000.0_rp, ' ms'

    if (NEKO_BCKND_DEVICE .eq. 1) then

      if (tstep .le. this%start_rl_tstep) then
        call spalding_initialize_device(u%x_d, v%x_d, w%x_d, this%ind_r_d, this%ind_s_d, this%ind_t_d, this%ind_e_d, &
        this%n_x%x_d, this%n_y%x_d, this%n_z%x_d, this%nu%x_d, this%h%x_d, &
        this%tau_x%x_d, this%tau_y%x_d, this%tau_z%x_d, this%n_nodes, u%Xh%lx, this%kappa, this%B, tstep, & 
        this%tau_true, &
        this%ui_l%x_d, this%vi_l%x_d, this%wi_l%x_d, this%normu_l%x_d, this%magu_l%x_d, this%vg_l%x_d, this%utau_l%x_d, &
        this%tau_old_l%x_d, this%tau_new_l%x_d, &
        this%l_star%x_d, this%u_plus%x_d, this%g_plus%x_d, this%h_plus%x_d, this%slope%x_d, this%intercept%x_d, this%dudy_d, &
        this%error_new%x_d, this%error_old%x_d, this%rel_error%x_d, &
        this%reward%x_d, this%total_reward%x_d, this%reward_out%x_d, this%base_reward%x_d, this%bonus_reward%x_d, &
        this%terminal%x_d, &
        this%state_d, this%action_d, &
        this%msk_d, this%reward_field%x_d, this%slope_field%x_d, this%intercept_field%x_d)

      else        
        ! ::::::> Getting the input state ::::::
        ! start_time = MPI_WTIME()
        call rlwm_compute_device(u%x_d, v%x_d, w%x_d, this%ind_r_d, this%ind_s_d, this%ind_t_d, this%ind_e_d, &
          this%n_x%x_d, this%n_y%x_d, this%n_z%x_d, this%nu%x_d, this%h%x_d, &
          this%tau_x%x_d, this%tau_y%x_d, this%tau_z%x_d, this%n_nodes, u%Xh%lx, this%kappa, this%B, tstep, &
          this%model_device, this%rb_device, this%start_rl_tstep, this%tsteps_rl, this%episode_length, this%n_epochs, & 
          this%tau_true, &
          this%ui_l%x_d, this%vi_l%x_d, this%wi_l%x_d, this%normu_l%x_d, this%magu_l%x_d, this%vg_l%x_d, this%utau_l%x_d, &
          this%tau_old_l%x_d, this%tau_new_l%x_d, &
          this%l_star%x_d, this%u_plus%x_d, this%g_plus%x_d, this%h_plus%x_d, this%slope%x_d, this%intercept%x_d, this%dudy_d, &
          this%error_new%x_d, this%error_old%x_d, this%rel_error%x_d, &
          this%reward%x_d, this%total_reward%x_d, this%reward_out%x_d, this%base_reward%x_d, this%bonus_reward%x_d, &
          this%terminal%x_d, &
          this%state_d, this%action_d, &
          this%msk_d, this%reward_field%x_d, this%slope_field%x_d, this%intercept_field%x_d)
        ! end_time = MPI_WTIME()
        ! if (pe_rank == 0) write(*, *) 'Time: Collect State (Sim. CUDA Kernel) = ', (end_time - start_time) * 1000.0_rp, ' ms'

        call this%get_global_state()
        call this%predict_global_action()
        call this%get_device_action()

        ! ::::::> Apply actuation to tau_w & collect rewards corresponding to this state transition ::::::
        ! start_time = MPI_WTIME()
        call rlwm_actuate_device(this%n_nodes, tstep, this%start_rl_tstep, this%tsteps_rl, this%episode_length, &
              this%action_d, this%tau_old_l%x_d, this%tau_new_l%x_d, this%utau_l%x_d, &
              this%tau_x%x_d, this%tau_y%x_d, this%tau_z%x_d, this%tau_true, &
              this%ui_l%x_d, this%vi_l%x_d, this%wi_l%x_d, this%magu_l%x_d, &
              this%error_new%x_d, this%error_old%x_d, this%rel_error%x_d, &
              this%reward%x_d, this%total_reward%x_d, this%base_reward%x_d, this%bonus_reward%x_d, &
              this%msk_d, this%reward_field%x_d)
        ! end_time = MPI_WTIME()
        ! if (pe_rank == 0) write(*, *) 'Time: Actuation (Sim. CUDA Kernel) = ', (end_time - start_time) * 1000.0_rp, ' ms'

        call this%get_global_reward()
        call this%get_global_terminal()
        if (mod(tstep - this%start_rl_tstep, this%tsteps_rl) .eq. 0) call this%update_buffer()
        ! call this%update_buffer()

        ! ::::::> Train after the buffer is full at the end of the episode ::::::
        train_starter = mod((real(tstep) - real(this%start_rl_tstep)) / real(this%tsteps_rl), real(this%episode_length))
        if (train_starter .eq. 0.0_rp) then
          call this%train_and_save()
          ! res = torchfort_rl_off_policy_evaluate(this%tf_key, this%state_older, this%action_older, reward_out)
        end if

      end if

    else
      call rlwm_compute_cpu(u%x, v%x, w%x, &
            this%ind_r, this%ind_s, this%ind_t, this%ind_e, &
            this%n_x%x, this%n_y%x, this%n_z%x, &
            this%nu%x, this%h%x, &
            this%tau_x%x, this%tau_y%x, this%tau_z%x, &
            this%n_nodes, u%Xh%lx, u%msh%nelv, &
            this%kappa, this%B, tstep, & 
            this%tf_key, this%yaml_path, this%log_dir, this%policy_method, this%phase, &
            this%model_device, this%rb_device, this%start_rl_tstep, this%tsteps_rl, this%n_epochs, this%tau_true, &
            this%ui_l%x, this%vi_l%x, this%wi_l%x, this%normu_l%x, this%magu_l%x, this%vg_l%x, this%utau_l%x, &
            this%tau_old_l%x, this%tau_new_l%x, &
            this%l_star%x, this%u_plus%x, this%g_plus%x, this%h_plus%x, this%slope%x, this%intercept%x, this%dudy, &
            this%error_new%x, this%error_old%x, this%rel_error%x, &
            this%reward%x, this%total_reward%x, this%reward_out%x, this%base_reward%x, this%bonus_reward%x, &
            this%terminal%x, &
            this%recvcounts, this%displs, this%total_agents, this%state, this%action, this%global_state, this%global_action, &
            this%episode, this%global_state_older, this%global_action_older, this%global_reward, this%global_terminal, &
            this%p_loss_val, this%q_loss_val, &
            this%msk, this%reward_field, this%slope_field, this%intercept_field)
    end if

  end subroutine rlwm_compute

  !=====================
  ! To get global_state
  !=====================
  subroutine rlwm_get_global_state(this)

    class(rlwm_t), intent(inout) :: this
    integer :: ierr 

    ! start_time = MPI_WTIME()  
    ! D2H on local state
    call device_memcpy(this%state, this%state_d, size(this%state), DEVICE_TO_HOST, sync = .true.)
    !> Gatherv to get global_state from local states
    call MPI_Gatherv(this%state, 2*this%n_nodes, MPI_DOUBLE_PRECISION, &
        this%global_state, 2*this%recvcounts, 2*this%displs, MPI_DOUBLE_PRECISION, &
        0, NEKO_COMM, ierr)
    if (pe_rank == 0) then
      write(*, '(A, A, I6, A, I6, 1X, I6, A, I6, A, F20.5)') &
        'global_state: ', 'ierr = ', ierr, ', shape = ', shape(this%global_state), & 
        ', size = ', size(this%global_state), ', sum = ', sum(this%global_state)
    end if
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: D2H + MPI_Gatherv for global_state = ', (end_time - start_time) * 1000.0_rp, ' ms'

  end subroutine rlwm_get_global_state

  !=================
  ! To get action_d
  !=================
  subroutine rlwm_get_device_action(this)

    class(rlwm_t), intent(inout) :: this
    integer :: ierr 

    ! start_time = MPI_WTIME()
    ! Scatterv to distribute global_action to local actions
    call MPI_Scatterv(this%global_action, this%recvcounts, this%displs, MPI_DOUBLE_PRECISION, &
                      this%action, this%n_nodes, MPI_DOUBLE_PRECISION, &
                      0, NEKO_COMM, ierr)        
    ! H2D copy of action array to device for GPU computation
    call device_memcpy(this%action, this%action_d, size(this%action), &
                       HOST_TO_DEVICE, sync = .true.) 
    if (pe_rank == 0) then
      write(*, '(A, A, I6, A, I6, 1X, I6, A, I6, A, F20.5)') &
        'global_action: ', 'ierr = ', ierr, ', shape = ', shape(this%global_action), & 
        ', size = ', size(this%global_action), ', sum = ', sum(this%global_action)
    end if
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: MPI_Scatterv + H2D on action = ', (end_time - start_time) * 1000.0_rp, ' ms'

  end subroutine rlwm_get_device_action

  !======================
  ! To get global_action
  !======================
  subroutine rlwm_predict_global_action(this)

    class(rlwm_t), intent(inout) :: this
    integer :: res
    real(kind=rp) :: start_time, end_time

    if (pe_rank .eq. 0) then
      ! start_time = MPI_WTIME()
      select case (trim(this%policy_method))
      case ("on-policy")
        ! res = torchfort_rl_on_policy_predict(this%tf_key, this%global_state, this%global_action)
        res = torchfort_rl_on_policy_predict_explore(this%tf_key, this%global_state, this%global_action)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
      case ("off-policy")
        ! res = torchfort_rl_off_policy_predict(this%tf_key, this%global_state, this%global_action)
        res = torchfort_rl_off_policy_predict_explore(this%tf_key, this%global_state, this%global_action)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
      end select
      ! end_time = MPI_WTIME()
      ! if (pe_rank == 0) write(*, *) 'Time: Predict Explore = ', (end_time - start_time) * 1000.0_rp, ' ms'
    end if

  end subroutine rlwm_predict_global_action

  !======================
  ! To get global_reward
  !======================
  subroutine rlwm_get_global_reward(this)
    
    class(rlwm_t), intent(inout) :: this
    integer :: ierr
    
    ! start_time = MPI_WTIME()
    ! D2H on local reward
    call device_memcpy(this%reward%x, this%reward%x_d, size(this%reward%x), DEVICE_TO_HOST, sync = .true.)
    ! Gatherv to get global_reward from local rewards
    call MPI_Gatherv(this%reward%x, this%n_nodes, MPI_DOUBLE_PRECISION, &
                     this%global_reward, this%recvcounts, this%displs, MPI_DOUBLE_PRECISION, &
                     0, NEKO_COMM, ierr)
    if (pe_rank == 0) then
      write(*, '(A, A, I6, A, I6, A, I6, A, F20.5)') &
      'global_reward: ', 'ierr = ', ierr, ', shape = ', shape(this%global_reward), & 
      ', size = ', size(this%global_reward), ', sum = ', sum(this%global_reward)
    end if
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: D2H + MPI_Gatherv for global_reward = ', (end_time - start_time) * 1000.0_rp, ' ms'

  end subroutine rlwm_get_global_reward

  !=======================
  ! To get global_terminal  
  !=======================
  subroutine rlwm_get_global_terminal(this)

    class(rlwm_t), intent(inout) :: this
    integer :: ierr

    ! start_time = MPI_WTIME()
    ! D2H on local terminal
    call device_memcpy(this%terminal%x, this%terminal%x_d, size(this%terminal%x), DEVICE_TO_HOST, sync = .true.)
    call MPI_Gatherv(this%terminal%x, this%n_nodes, MPI_DOUBLE_PRECISION, &
            this%global_terminal, this%recvcounts, this%displs, MPI_DOUBLE_PRECISION, &
            0, NEKO_COMM, ierr)
    if (pe_rank == 0) then
      write(*, '(A, A, I6, A, I6, A, I6, A, F20.5)') &
      'global_terminal: ', 'ierr = ', ierr, ', shape = ', shape(this%global_terminal), & 
      ', size = ', size(this%global_terminal), ', sum = ', sum(this%global_terminal)
    end if
    ! end_time = MPI_WTIME()
    ! if (pe_rank == 0) write(*, *) 'Time: D2H + MPI_Gatherv for global_terminal = ', (end_time - start_time) * 1000.0_rp, ' ms'

  end subroutine rlwm_get_global_terminal

  !===============
  ! Update Buffer
  !===============
  subroutine rlwm_update_buffer(this)

    class(rlwm_t), intent(inout) :: this
    integer :: res
    real(kind=rp) :: start_time, end_time

    if (pe_rank .eq. 0) then 
      ! start_time = MPI_WTIME()
      select case (trim(this%policy_method))
      case ("on-policy")
        res = torchfort_rl_on_policy_update_rollout_buffer(this%tf_key, & 
        this%global_state_older, this%global_action_older, this%global_reward, this%global_terminal)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
      case ("off-policy")
        res = torchfort_rl_off_policy_update_replay_buffer(this%tf_key, & 
        this%global_state_older, this%global_action_older, this%global_state, this%global_reward, this%global_terminal)
        if (res /= TORCHFORT_RESULT_SUCCESS) stop
      end select
      ! end_time = MPI_WTIME()
      ! if (pe_rank == 0) write(*, *) 'Time: Update Replay Buffer = ', (end_time - start_time) * 1000.0_rp, ' ms'
    end if

  end subroutine rlwm_update_buffer

  !===================
  ! Training & Saving 
  !===================
  subroutine rlwm_train_and_save(this)
    
    class(rlwm_t), intent(inout) :: this
    integer :: ierr, res
    logical :: is_ready
    integer :: epoch

    select case (trim(this%policy_method))
    case ("on-policy")
      res = torchfort_rl_on_policy_is_ready(this%tf_key, is_ready)
      do epoch = 1, this%n_epochs
        if (is_ready) then
          res = torchfort_rl_on_policy_train_step(this%tf_key, this%p_loss_val, this%q_loss_val)
          print *, "Epoch = ", epoch, " , p_loss = ", this%p_loss_val, " , q_loss = ", this%q_loss_val
        end if
      end do
      res = torchfort_rl_on_policy_save_checkpoint(this%tf_key, this%log_dir)
      if (res /= TORCHFORT_RESULT_SUCCESS) stop
    case ("off-policy")
      ! start_time = MPI_WTIME()
      res = torchfort_rl_off_policy_save_checkpoint(this%tf_key, this%log_dir)
      if (res /= TORCHFORT_RESULT_SUCCESS) stop
      ! end_time = MPI_WTIME()
      ! if (pe_rank == 0) write(*, *) 'Time: Saving Checkpoint = ', (end_time - start_time) * 1000.0_rp, ' ms'
      ! start_time = MPI_WTIME()
      res = torchfort_rl_off_policy_is_ready(this%tf_key, is_ready)
      ! end_time = MPI_WTIME()
      ! if (pe_rank == 0) write(*, *) 'Time: Check if ready for training = ', (end_time - start_time) * 1000.0_rp, ' ms'
      print *, "is_ready = ", is_ready
      ! start_time = MPI_WTIME()
      do epoch = 1, this%n_epochs
        print *, "Epoch = ", epoch
        if (is_ready) then
          res = torchfort_rl_off_policy_train_step(this%tf_key, this%p_loss_val, this%q_loss_val)
        end if
      end do
      ! end_time = MPI_WTIME()
      ! if (pe_rank == 0) write(*, *) 'Time: Train Step Loop (all epochs) = ', (end_time - start_time) * 1000.0_rp, ' ms'
    end select

  end subroutine rlwm_train_and_save
  
end module rlwm
