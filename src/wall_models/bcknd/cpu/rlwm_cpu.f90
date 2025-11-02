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
!> Implements the CPU kernel for the `rlwm_t` type.
module rlwm_cpu
  use num_types, only : rp
  use logger, only : neko_log, NEKO_LOG_DEBUG, LOG_SIZE
  
  !> TorchFort =====================================================================
  use comm, only : pe_size, pe_rank, NEKO_COMM
  use mpi_f08, only : MPI_Gatherv, MPI_Scatterv, MPI_DOUBLE_PRECISION, MPI_IN_PLACE
  use torchfort
  use num_types, only : sp
  use field, only : field_t
  !=================================================================================
  
  implicit none
  private

  public :: rlwm_compute_cpu

contains
  !> Compute the wall shear stress on cpu using rlwm's model.
  !! @param t The time value.
  !! @param tstep The current time-step.
  subroutine rlwm_compute_cpu(u, v, w, ind_r, ind_s, ind_t, ind_e, &
       n_x, n_y, n_z, nu, h, tau_x, tau_y, tau_z, n_nodes, lx, nelv, &
       kappa, B, tstep, &
	   tf_key, yaml_path, log_dir, policy_method, phase, & 
	   model_device, rb_device, start_rl_tstep, tsteps_rl, n_epochs, tau_true, &
	   ui_l, vi_l, wi_l, normu_l, magu_l, vg_l, utau_l, & 
	   tau_old_l, tau_new_l, &
	   l_star, u_plus, g_plus, h_plus, slope, intercept, dudy, &
	   error_new, error_old, rel_error, &
	   reward, total_reward, reward_out, base_reward, bonus_reward, &
	   terminal, &
	   recvcounts, displs, total_agents, state, action, global_state, global_action, &
	   episode, global_state_older, global_action_older, global_reward, global_terminal, &
	   p_loss_val, q_loss_val, &
	   msk, reward_field, slope_field, intercept_field)
	   
    integer, intent(in) :: n_nodes, lx, nelv, tstep
    real(kind=rp), dimension(lx, lx, lx, nelv), intent(in) :: u, v, w
    integer, intent(in), dimension(n_nodes) :: ind_r, ind_s, ind_t, ind_e
    real(kind=rp), dimension(n_nodes), intent(in) :: n_x, n_y, n_z, h, nu
    real(kind=rp), dimension(n_nodes), intent(inout) :: tau_x, tau_y, tau_z
    real(kind=rp), intent(in) :: kappa, B
    integer :: i
    real(kind=rp) :: ui, vi, wi, magu, utau, normu, guess
	
	!> TorchFort ******************************************************************************************************************
	!> JSON INPUTS 
    character(len=*), intent(in) :: tf_key, yaml_path, log_dir, policy_method, phase
    integer, intent(in) :: model_device, rb_device, start_rl_tstep, tsteps_rl, n_epochs
    real(kind=rp), intent(in) :: tau_true
	!> 1D Arrays
	real(kind=rp), dimension(n_nodes), intent(inout) :: ui_l, vi_l, wi_l, normu_l, magu_l, vg_l, utau_l, & 
														tau_old_l, tau_new_l, &
														l_star, u_plus, g_plus, h_plus, slope, intercept, &
														error_new, error_old, rel_error, &
														reward, total_reward, reward_out, base_reward, bonus_reward, &
														terminal
	integer :: total_agents, episode
	integer, intent(in) :: msk(0:)
	type(field_t), pointer, intent(inout) :: reward_field, slope_field, intercept_field
	!> 4D Arrays
	real(kind=rp), dimension(lx, lx, lx, nelv), intent(inout) :: dudy
	!> 1D Integer Arrays
	integer, dimension(pe_size), intent(inout) :: recvcounts, displs
	!> 1D Real Arrays
	real(kind=rp), dimension(total_agents), intent(inout) :: global_reward, global_terminal	
	!> 2D Arrays
	real(kind=rp), dimension(2, n_nodes), intent(inout) :: state
	real(kind=rp), dimension(1, n_nodes), intent(inout) :: action
	real(kind=rp), dimension(2, total_agents), intent(inout) :: global_state, global_state_older
	real(kind=rp), dimension(1, total_agents), intent(inout) :: global_action, global_action_older
	!> Single precision loss values
	real(kind=sp) :: p_loss_val, q_loss_val
	!> Local Variables
	integer :: epoch, ierr, res
	logical :: is_ready = .false.
	!******************************************************************************************************************************
	
    if (tstep .eq. 1 .and. pe_rank .eq. 0) then
      print *, ""
      print *, "------------------------------------------------------"
      print *, "|      Verifying RLWM parameters from JSON file:     |"
      print *, "------------------------------------------------------"
      print '(" 01. tf_key            : ", A)', tf_key
      print '(" 02. yaml_path         : ", A)', yaml_path
      print '(" 03. log_dir           : ", A)', log_dir
	  print '(" 04. policy_method     : ", A)', policy_method
	  print '(" 05. phase             : ", A)', phase
      print '(" 06. model_device      : ", I0)', model_device
      print '(" 07. rb_device         : ", I0)', rb_device
      print '(" 08. tau_true          : ", F0.6)', tau_true	  
	  print '(" 09. start_rl_tstep : ", I0)', start_rl_tstep
	  print '(" 10. tsteps_rl         : ", I0)', tsteps_rl
	  print '(" 11. n_epochs          : ", I0)', n_epochs
      print *, "------------------------------------------------------"
      print *, ""
    end if

	do i=1, n_nodes
	
       ! Sample the velocity
       ui = u(ind_r(i), ind_s(i), ind_t(i), ind_e(i))
       vi = v(ind_r(i), ind_s(i), ind_t(i), ind_e(i))
       wi = w(ind_r(i), ind_s(i), ind_t(i), ind_e(i))

       ! Project on tangential direction
       normu = ui * n_x(i) + vi * n_y(i) + wi * n_z(i)

       ui = ui - normu * n_x(i)
       vi = vi - normu * n_y(i)
       wi = wi - normu * n_z(i)

       magu = sqrt(ui**2 + vi**2 + wi**2)
	   
	   ! Arrays to use in the do loop for action
	   ui_l(i) = ui
	   vi_l(i) = vi
	   wi_l(i) = wi
	   normu_l(i) = normu
	   magu_l(i) = magu

       ! Get initial guess for Newton solver
       if (tstep .eq. 1) then
		 ! Guess velocity
         vg_l(i) = sqrt(magu_l(i) * nu(i) / h(i))
		 ! Friction velocity
		 utau_l(i) =  solve_cpu(magu_l(i), h(i), vg_l(i), nu(i), kappa, B)
		 ! Shear Stress resulting from the computed utau
		 tau_old_l(i) = utau_l(i)**2
		 ! Distribute according to the velocity vector
         tau_x(i) = -utau_l(i)**2 * ui_l(i) / magu_l(i)
         tau_y(i) = -utau_l(i)**2 * vi_l(i) / magu_l(i)
         tau_z(i) = -utau_l(i)**2 * wi_l(i) / magu_l(i)	
		 ! DEBUG
		 ! call print_debug_info(i, 99, 103, n_nodes, &
		 ! ui_l, vi_l, wi_l, normu_l, magu_l, h, &
		 ! vg_l, utau_l, tau_x, tau_z, tau_old_l, tau_new_l, &
		 ! l_star, u_plus, g_plus, h_plus, state, &
		 ! reward, total_reward, reward_out, base_reward, bonus_reward)	
		 
       else if (tstep .le. start_rl_tstep) then
	     ! Magnitude of Shear Stress
         tau_old_l(i) = sqrt(tau_x(i)**2 + tau_y(i)**2 + tau_z(i)**2)
		 ! Guess Velocity
         vg_l(i) = sqrt(tau_old_l(i))
		 ! Friction velocity
		 utau_l(i) =  solve_cpu(magu_l(i), h(i), vg_l(i), nu(i), kappa, B)
		 ! Shear Stress resulting from the computed utau
		 tau_new_l(i) = utau_l(i)**2
		 ! Calculation of Reward
		 call calculate_reward(i, n_nodes, tau_true, tau_old_l, tau_new_l, error_new, error_old, rel_error, &
							   reward, total_reward, reward_out, base_reward, bonus_reward)	
		 ! Copy reward value to the field
         reward_field%x(msk(i),1,1,1) = reward(i)					   
		 ! Distribute according to the velocity vector
         tau_x(i) = -utau_l(i)**2 * ui_l(i) / magu_l(i)
         tau_y(i) = -utau_l(i)**2 * vi_l(i) / magu_l(i)
         tau_z(i) = -utau_l(i)**2 * wi_l(i) / magu_l(i)
		 ! DEBUG
		 ! call print_debug_info(i, 99, 103, n_nodes, &
		 ! ui_l, vi_l, wi_l, normu_l, magu_l, h, &
		 ! vg_l, utau_l, tau_x, tau_z, tau_old_l, tau_new_l, &
		 ! l_star, u_plus, g_plus, h_plus, state, &
		 ! reward, total_reward, reward_out, base_reward, bonus_reward)
		 
	   else 
	   	 ! Magnitude of Shear Stress
         tau_old_l(i) = sqrt(tau_x(i)**2 + tau_y(i)**2 + tau_z(i)**2) 
		 ! Friction velocity		 
		 utau_l(i) = sqrt(tau_old_l(i))
		 ! Guess Velocity
		 vg_l(i) = utau_l(i)
		 
       end if
	   
	   ! Normalization w.r.t. viscous scales (nu, utau)
       l_star(i) = nu(i) / (utau_l(i) + 1e-6)
       u_plus(i) = magu_l(i) / (utau_l(i) + 1e-6)
       g_plus(i) = dudy(ind_r(i), ind_s(i), ind_t(i), ind_e(i)) / ( (utau_l(i) + 1e-6) / l_star(i))
       h_plus(i) = h(i) / (l_star(i) + 1e-6)

       ! Changing to normalized states
	   slope(i) = h_plus(i) * g_plus(i) - (log(h_plus(i))/kappa)
	   intercept(i) = u_plus(i) - (log(h_plus(i))/kappa)
	   
	   ! Writing to 2D state array
	   state(1,i) = slope(i)
       state(2,i) = intercept(i)
	   
	   ! Field for visualization
	   slope_field%x(msk(i),1,1,1) = slope(i)
	   intercept_field%x(msk(i),1,1,1) = intercept(i)

	end do
	
	!> MPI_Gatherv for global_state -----------------------------------------------------------------------------------------------
	call MPI_Gatherv(state, 2*n_nodes, MPI_DOUBLE_PRECISION, &
					 global_state, 2*recvcounts, 2*displs, MPI_DOUBLE_PRECISION, &
					 0, NEKO_COMM, ierr)
	! if (pe_rank==0) then
		! print *, "ierr from MPI_Gatherv = ", ierr
		! print *, "shape(global_state): ", shape(global_state)
		! print *, "size(global_state): ", size(global_state)
		! print *
	! end if
	! -----------------------------------------------------------------------------------------------------------------------------
	
	!> Predict actions by doing a forward pass through the policy network ---------------------------------------------------------
	if (pe_rank .eq. 0) then
		select case (trim(policy_method))
		case ("on-policy")
			! res = torchfort_rl_on_policy_predict(tf_key, global_state, global_action)
			res = torchfort_rl_on_policy_predict_explore(tf_key, global_state, global_action)
			if (res /= TORCHFORT_RESULT_SUCCESS) stop
		case ("off-policy")
			res = torchfort_rl_off_policy_predict(tf_key, global_state, global_action)
			! res = torchfort_rl_off_policy_predict_explore(tf_key, global_state, global_action)
			if (res /= TORCHFORT_RESULT_SUCCESS) stop
		end select
	end if
	! -----------------------------------------------------------------------------------------------------------------------------
	
	!> MPI_Scatterv for global_action ---------------------------------------------------------------------------------------------
	call MPI_Scatterv(global_action, recvcounts, displs, MPI_DOUBLE_PRECISION, &
					 action, n_nodes, MPI_DOUBLE_PRECISION, &
					 0, NEKO_COMM, ierr)
	! if (pe_rank==0) then
		! print *, "ierr from MPI_Scatterv = ", ierr
		! print *, "shape(global_action): ", shape(global_action)
		! print *, "size(global_action): ", size(global_action)
	! end if
	! -----------------------------------------------------------------------------------------------------------------------------

	! Start using RL for actuation & training
    if (tstep .gt. start_rl_tstep) then
	
		! Apply actuation to wall shear stress
		do i=1, n_nodes
		
		  if (mod(tstep, tsteps_rl) == 0) then
		     tau_new_l(i) = tau_old_l(i) * action(1,i)
		  else
             tau_new_l(i) = tau_old_l(i) + (tau_new_l(i) - tau_old_l(i)) * (mod(tstep, tsteps_rl) / tsteps_rl)
		  end if
		  
		  ! Friction velocity based on new wall shear stress
		  utau_l(i) = sqrt(tau_new_l(i))
		  
		  ! Distribute according to the velocity vector
		  tau_x(i) = -utau_l(i)**2 * ui_l(i) / magu_l(i)
		  tau_y(i) = -utau_l(i)**2 * vi_l(i) / magu_l(i)
		  tau_z(i) = -utau_l(i)**2 * wi_l(i) / magu_l(i)
		  
		  ! Calculation of Reward
		  call calculate_reward(i, n_nodes, tau_true, tau_old_l, tau_new_l, error_new, error_old, rel_error, &
							    reward, total_reward, reward_out, base_reward, bonus_reward)
		  
		  ! DEBUG
		  ! call print_debug_info(i, 99, 103, n_nodes, &
		                        ! ui_l, vi_l, wi_l, normu_l, magu_l, h, &
		                        ! vg_l, utau_l, tau_x, tau_z, tau_old_l, tau_new_l, &
		                        ! l_star, u_plus, g_plus, h_plus, state, &
		                        ! reward, total_reward, reward_out, base_reward, bonus_reward)					
		  
		end do ! End of action do loop

		if ((trim(phase) .eq. 'training') .and. (pe_rank .eq. 0)) then
		
			! Update rollout buffer (or) replay buffer
			! if (pe_rank .eq. 0) then
				select case (trim(policy_method))
				case ("on-policy")
					res = torchfort_rl_on_policy_update_rollout_buffer(tf_key, & 
						  global_state_older, global_action_older, global_reward, global_terminal)
					if (res /= TORCHFORT_RESULT_SUCCESS) stop
				case ("off-policy")
					res = torchfort_rl_off_policy_update_replay_buffer(tf_key, & 
						  global_state_older, global_action_older, global_state, global_reward, global_terminal)
					if (res /= TORCHFORT_RESULT_SUCCESS) stop
				end select
			! end if

			! Training & Saving
			if (mod(tstep, tsteps_rl) == 0) then
				select case (trim(policy_method))
				case ("on-policy")
					res = torchfort_rl_on_policy_is_ready(tf_key, is_ready)
					do epoch = 1, n_epochs
						if (is_ready) then
							res = torchfort_rl_on_policy_train_step(tf_key, p_loss_val, q_loss_val)
						end if
					end do
					res = torchfort_rl_on_policy_save_checkpoint(tf_key, log_dir)
					if (res /= TORCHFORT_RESULT_SUCCESS) stop
				case ("off-policy")
					res = torchfort_rl_off_policy_is_ready(tf_key, is_ready)
					do epoch = 1, n_epochs
						if (is_ready) then
							res = torchfort_rl_off_policy_train_step(tf_key, p_loss_val, q_loss_val)
						end if
					end do
					res = torchfort_rl_off_policy_save_checkpoint(tf_key, log_dir)
					if (res /= TORCHFORT_RESULT_SUCCESS) stop
				end select
			end if
			
		end if
		
		! res = torchfort_rl_off_policy_evaluate(this%tf_key, this%state_older, this%action_older, reward_out)
    end if

  end subroutine rlwm_compute_cpu
  
!==================================================================================================================================
!> Calculates the instantaneous reward for agent 'i'
!==================================================================================================================================
subroutine calculate_reward(i, n_nodes, tau_true, tau_old_l, tau_new_l, error_new, error_old, rel_error, &
							reward, total_reward, reward_out, base_reward, bonus_reward)
  
  integer :: i, n_nodes
  real(kind=rp), intent(in) :: tau_true
  real(kind=rp), dimension(n_nodes), intent(inout) :: tau_old_l, tau_new_l, error_new, error_old, rel_error, &
													  reward, total_reward, reward_out, base_reward, bonus_reward
  
  ! error_new(i) = abs(tau_true - tau_new_l(i))
  ! error_old(i) = abs(tau_true - tau_old_l(i))
  ! base_reward(i) = (error_new(i) - error_old(i)) / tau_true
  
  ! rel_error(i) = error_new(i) / tau_true
  ! if (rel_error(i) < 0.01_rp) then
	 ! bonus_reward(i) = 1.0_rp - rel_error(i)
  ! else
	 ! bonus_reward(i) = 0.0_rp
  ! end if

  ! ! Reward collected by agent 'i' at 'tstep'
  ! reward(i) = base_reward(i) + bonus_reward(i)
  
  ! ! Reward collected by agent 'i' in one episode consisting of 'tsteps_rl' trajectories
  ! total_reward(i) = total_reward(i) + reward(i)
  
  reward(i) = abs(tau_new_l(i) - tau_true) / tau_true
  total_reward(i) = total_reward(i) + reward(i)

end subroutine calculate_reward

!==================================================================================================================================
!> Print all debug info agent 'i'
!==================================================================================================================================
subroutine print_debug_info(i, a, b, n_nodes, &
							ui_l, vi_l, wi_l, normu_l, magu_l, h, &
							vg_l, utau_l, tau_x, tau_z, tau_old_l, tau_new_l, &
							l_star, u_plus, g_plus, h_plus, state, &
							reward, total_reward, reward_out, base_reward, bonus_reward)

    integer :: i, a, b, n_nodes
	real(kind=rp), dimension(n_nodes), intent(in) :: h
	real(kind=rp), dimension(n_nodes), intent(inout) :: ui_l, vi_l, wi_l, normu_l, magu_l, & 
														vg_l, utau_l, tau_x, tau_z, tau_old_l, tau_new_l, &
														l_star, u_plus, g_plus, h_plus, &
														reward, total_reward, reward_out, base_reward, bonus_reward
	real(kind=rp), dimension(2, n_nodes), intent(inout) :: state
	
    if (pe_rank == 0) then
        if (i >= a .and. i <= b) then
            if (i == a) then
                write(*, *) '======================================================================================================& 
				=============================================='
                write(*, '(A6, A20, A20, A20, A20, A20, A20)') &
                    'i', 'ui_l', 'vi_l', 'wi_l', 'normu_l', 'magu_l', 'h'
                write(*, '(A6, A20, A20, A20, A20, A20, A20)') &
                    ' ', 'vg_l', 'utau_l', 'tau_x', 'tau_z', 'tau_old_l', 'tau_new_l'
                write(*, '(A6, A20, A20, A20, A20, A20, A20)') &
                    ' ', 'state_1', 'state_2', 'l*', 'u+', 'g+', 'h+'
                write(*, '(A6, A20, A20, A20, A20, A20)') &
                    ' ', 'reward', 'total_reward', 'reward_out', 'base_reward', 'bonus_reward'
                write(*, *) '------------------------------------------------------------------------------------------------------&
				----------------------------------------------'
            end if

            write(*, '(I6, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5)') &
                i, ui_l(i), vi_l(i), wi_l(i), normu_l(i), magu_l(i), h(i)
            write(*, '(A6, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5)') &
                ' ', vg_l(i), utau_l(i), tau_x(i), tau_z(i), tau_old_l(i), tau_new_l(i)
            write(*, '(A6, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5)') &
                ' ', state(1,i), state(2,i), l_star(i), u_plus(i), g_plus(i), h_plus(i)
            write(*, '(A6, ES20.5, ES20.5, ES20.5, ES20.5, ES20.5)') &
                ' ', reward(i), total_reward(i), reward_out(i), base_reward(i), bonus_reward(i)
            write(*,*)
        end if
    end if

end subroutine print_debug_info

  !> Newton solver for the algebraic equation defined by the law on cpu.
  !! @param u The velocity value.
  !! @param y The wall-normal distance.
  !! @param guess Initial guess.
  !! @param nu The molecular kinematic viscosity.
  !! @param kappa The von Karman constant.
  !! @param B The log-law intercept.
  function solve_cpu(u, y, guess, nu, kappa, B) result(utau)
    real(kind=rp), intent(in) :: u
    real(kind=rp), intent(in) :: y
    real(kind=rp), intent(in) :: guess
    real(kind=rp), intent(in) :: nu, kappa, B
    real(kind=rp) :: yp, up, utau
    real(kind=rp) :: error, f, df, old
    integer :: niter, k, maxiter
    character(len=LOG_SIZE) :: log_msg

    utau = guess

    maxiter = 100

    do k=1, maxiter
       up = u / utau
       yp = y * utau / nu
       niter = k
       old = utau

       ! Evaluate function and its derivative
       f = (up + exp(-kappa*B)* &
            (exp(kappa*up) - 1.0_rp - kappa*up - 0.5_rp*(kappa*up)**2 - &
            1.0_rp/6*(kappa*up)**3) - yp)

       df = (-y / nu - u/utau**2 - kappa*up/utau*exp(-kappa*B) * &
            (exp(kappa*up) - 1 - kappa*up - 0.5*(kappa*up)**2))

       ! Update solution
       utau = utau - f / df

       error = abs((old - utau)/old)

       if (error < 1e-3) then
          exit
       endif

    enddo

    if (niter .eq. maxiter) then
       write(log_msg, *) "Newton not converged", error, f, utau, old, guess
       call neko_log%message(log_msg, NEKO_LOG_DEBUG)
    end if
  end function solve_cpu
end module rlwm_cpu
